import json
import os
import time
import hashlib
import boto3
import numpy as np
import faiss

# ── clients (outside handler = reused on warm starts) ─────────────
s3       = boto3.client("s3",              region_name="us-east-1")
bedrock  = boto3.client("bedrock-runtime", region_name="us-east-1")
dynamo   = boto3.resource("dynamodb",      region_name="us-east-1")

S3_BUCKET      = os.environ["S3_BUCKET"]       # itvidyarthi-rag-store
DYNAMODB_TABLE = os.environ["DYNAMODB_TABLE"]  # rag-response-cache

table = dynamo.Table(DYNAMODB_TABLE)

# ── global cache: persists across warm invocations ────────────────
_faiss_index = None
_metadata    = None


# ── 1. load index from s3 into /tmp (cold start only) ─────────────
def load_index():
    global _faiss_index, _metadata

    if _faiss_index is not None:
        print("Warm start — using cached index")
        return

    print("Cold start — downloading index from S3")
    s3.download_file(S3_BUCKET, "index/faiss.index", "/tmp/faiss.index")
    s3.download_file(S3_BUCKET, "index/metadata.json", "/tmp/metadata.json")

    _faiss_index = faiss.read_index("/tmp/faiss.index")

    with open("/tmp/metadata.json", "r") as f:
        _metadata = json.load(f)

    print(f"Index loaded — {_faiss_index.ntotal} vectors, {len(_metadata)} chunks")


# ── 2. embed query via bedrock titan ──────────────────────────────
def embed_query(text):
    body = json.dumps({"inputText": text})
    resp = bedrock.invoke_model(
        modelId     = "amazon.titan-embed-text-v2:0",
        body        = body,
        contentType = "application/json",
        accept      = "application/json",
    )
    result = json.loads(resp["body"].read())
    return np.array(result["embedding"], dtype="float32").reshape(1, -1)


# ── 3. retrieve top-k chunks from faiss ───────────────────────────
def retrieve_chunks(query_vector, k=5):
    distances, indices = _faiss_index.search(query_vector, k)
    chunks = []
    for idx in indices[0]:
        if idx == -1:
            continue
        chunks.append(_metadata[idx])
    return chunks


# ── 4. check dynamodb cache ───────────────────────────────────────
def check_cache(query_hash):
    try:
        resp = table.get_item(Key={"query_hash": query_hash})
        item = resp.get("Item")
        if item and int(item.get("expires_at", 0)) > int(time.time()):
            print("Cache hit")
            return item.get("response")
    except Exception as e:
        print(f"Cache check failed: {e}")
    return None


# ── 5. write to dynamodb cache ────────────────────────────────────
def write_cache(query_hash, response):
    try:
        table.put_item(Item={
            "query_hash": query_hash,
            "response":   response,
            "expires_at": int(time.time()) + 86400,  # 24h TTL
        })
    except Exception as e:
        print(f"Cache write failed: {e}")


# ── 6. build prompt from chunks ───────────────────────────────────
def build_prompt(query, chunks, mode):
    context = ""
    for i, chunk in enumerate(chunks):
        context += (
            f"[{i+1}] {chunk['chunk_text']}\n"
            f"    Source: {chunk['title']} — {chunk['url']}\n\n"
        )

    if mode == "quiz":
        return (
            f"Using ONLY the context below, generate exactly 5 multiple choice "
            f"questions about: {query}\n\n"
            f"Format each question exactly like this:\n"
            f"Q1. [question text]\n"
            f"A) [option]\nB) [option]\nC) [option]\nD) [option]\n"
            f"Answer: [correct letter]\n\n"
            f"Context:\n{context}"
        )
    else:
        return (
            f"Using ONLY the context below, answer this question: {query}\n\n"
            f"Rules:\n"
            f"- Answer in 2-3 paragraphs\n"
            f"- Cite sources using [1], [2] etc. inline\n"
            f"- End with a Sources section listing each cited URL\n"
            f"- If the context doesn't contain enough info, say so\n\n"
            f"Context:\n{context}"
        )


# ── 7. call amazon nova lite via converse api ─────────────────────
def call_nova(prompt):
    resp = bedrock.converse(
        modelId  = "amazon.nova-lite-v1:0",
        messages = [
            {
                "role":    "user",
                "content": [{"text": prompt}],
            }
        ],
        system = [
            {
                "text": (
                    "You are a current affairs assistant for competitive exam "
                    "students in India (UPSC, GATE, placement prep). "
                    "Answer strictly using the provided context only. "
                    "Never make up facts. Always cite your sources."
                )
            }
        ],
        inferenceConfig = {
            "maxTokens":   1024,
            "temperature": 0.3,
        },
    )
    return resp["output"]["message"]["content"][0]["text"]


# ── 8. extract source urls from retrieved chunks ──────────────────
def extract_sources(chunks):
    seen = set()
    sources = []
    for chunk in chunks:
        url = chunk.get("url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append({
                "title": chunk.get("title", ""),
                "url":   url,
                "date":  chunk.get("date", ""),
            })
    return sources


# ── main handler ──────────────────────────────────────────────────
def handler(event, context):
    # -- parse API Gateway proxy event
    try:
        body  = json.loads(event.get("body") or "{}")
        query = body.get("query", "").strip()
        mode  = body.get("mode", "answer").strip().lower()
    except Exception:
        return {
            "statusCode": 400,
            "headers":    cors_headers(),
            "body":       json.dumps({"error": "Invalid request body"}),
        }

    if not query:
        return {
            "statusCode": 400,
            "headers":    cors_headers(),
            "body":       json.dumps({"error": "query field is required"}),
        }

    # -- load faiss index (skipped on warm starts)
    load_index()

    # -- check cache first
    cache_key = hashlib.md5(f"{query}:{mode}".encode()).hexdigest()
    cached    = check_cache(cache_key)
    if cached:
        return {
            "statusCode": 200,
            "headers":    cors_headers(),
            "body":       cached,
        }

    # -- embed + retrieve
    query_vector = embed_query(query)
    chunks       = retrieve_chunks(query_vector, k=5)
    sources      = extract_sources(chunks)

    # -- build prompt + call nova
    prompt = build_prompt(query, chunks, mode)
    answer = call_nova(prompt)

    # -- assemble response
    result = json.dumps({
        "answer":  answer,
        "sources": sources,
        "mode":    mode,
        "cached":  False,
    })

    # -- write to cache
    write_cache(cache_key, result)

    return {
        "statusCode": 200,
        "headers":    cors_headers(),
        "body":       result,
    }


def cors_headers():
    return {
        "Content-Type":                "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }