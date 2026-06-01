import json
import os
import io
import pickle
import hashlib
import datetime
import boto3
import numpy as np
import faiss
import urllib.request
import urllib.parse

# ── clients (initialised outside handler = reused on warm starts) ──
s3        = boto3.client("s3",              region_name="us-east-1")
bedrock   = boto3.client("bedrock-runtime", region_name="us-east-1")
sm        = boto3.client("secretsmanager",  region_name="us-east-1")

S3_BUCKET   = os.environ["S3_BUCKET"]    # itvidyarthi-rag-store
SECRET_NAME = os.environ["SECRET_NAME"]  # itvidyarthi/newsapi-key

TOPICS = [
    "Global Geopolitics",
    "Artificial Intelligence",
    "science and technology",
    "international relations India",
    "Computer Science trending",
    "Generative AI",
]

CHUNK_SIZE = 300   # characters per chunk (≈ 60-80 tokens)
EMBED_DIM  = 1024  # Titan Text Embeddings V2 output dimension


# ── 1. fetch newsapi key from secrets manager ──────────────────────
def get_api_key():
    resp = sm.get_secret_value(SecretId=SECRET_NAME)
    return json.loads(resp["SecretString"])["api_key"]


# ── 2. fetch articles from newsapi ────────────────────────────────
def fetch_articles(api_key):
    articles = []
    for topic in TOPICS:
        query   = urllib.parse.quote(topic)
        url     = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}&language=en&sortBy=publishedAt"
            f"&pageSize=10&apiKey={api_key}"
        )
        req  = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        for a in data.get("articles", []):
            # skip articles with no usable content
            if not a.get("title") or not a.get("description"):
                continue
            articles.append({
                "title":       a["title"],
                "url":         a["url"],
                "date":        a.get("publishedAt", "")[:10],
                "source":      a.get("source", {}).get("name", ""),
                "content":     (a.get("description") or "")
                               + " " +
                               (a.get("content")     or ""),
            })
    print(f"Fetched {len(articles)} articles across {len(TOPICS)} topics")
    return articles


# ── 3. chunk article text ─────────────────────────────────────────
def chunk_text(text, size=CHUNK_SIZE):
    words  = text.split()
    chunks = []
    current = []
    length  = 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= size:
            chunks.append(" ".join(current))
            current = []
            length  = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── 4. embed a single chunk via bedrock titan ─────────────────────
def embed(text):
    body = json.dumps({"inputText": text})
    resp = bedrock.invoke_model(
        modelId     = "amazon.titan-embed-text-v2:0",
        body        = body,
        contentType = "application/json",
        accept      = "application/json",
    )
    result = json.loads(resp["body"].read())
    return np.array(result["embedding"], dtype="float32")


# ── 5. build faiss index from all chunks ──────────────────────────
def build_index(all_vectors):
    matrix = np.vstack(all_vectors).astype("float32")
    index  = faiss.IndexFlatL2(EMBED_DIM)
    index.add(matrix)
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


# ── 6. save index + metadata + raw articles to s3 ────────────────
def save_to_s3(index, metadata, articles):
    today = datetime.date.today().isoformat()   # e.g. 2026-06-01

    # serialise faiss index to bytes
    buf = io.BytesIO()
    faiss.write_index(index, "/tmp/faiss.index")
    with open("/tmp/faiss.index", "rb") as f:
        index_bytes = f.read()

    s3.put_object(
        Bucket = S3_BUCKET,
        Key    = "index/faiss.index",
        Body   = index_bytes,
    )
    s3.put_object(
        Bucket      = S3_BUCKET,
        Key         = "index/metadata.json",
        Body        = json.dumps(metadata),
        ContentType = "application/json",
    )
    s3.put_object(
        Bucket      = S3_BUCKET,
        Key         = f"raw/{today}.json",
        Body        = json.dumps(articles),
        ContentType = "application/json",
    )
    print(f"Saved index ({len(metadata)} chunks) and raw/{today}.json to S3")


# ── main handler ──────────────────────────────────────────────────
def handler(event, context):
    api_key  = get_api_key()
    articles = fetch_articles(api_key)

    all_vectors = []
    metadata    = []          # parallel list: index i → chunk info

    for article in articles:
        chunks = chunk_text(article["content"])
        for chunk in chunks:
            if len(chunk.strip()) < 30:   # skip tiny fragments
                continue
            vector = embed(chunk)
            all_vectors.append(vector)
            metadata.append({
                "chunk_text": chunk,
                "title":      article["title"],
                "url":        article["url"],
                "date":       article["date"],
                "source":     article["source"],
            })

    if not all_vectors:
        return {"statusCode": 500, "body": "No vectors generated"}

    index = build_index(all_vectors)
    save_to_s3(index, metadata, articles)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "chunks_indexed": len(metadata),
            "articles_fetched": len(articles),
        })
    }