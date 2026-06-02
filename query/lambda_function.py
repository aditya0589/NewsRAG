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

TOPICS = [
    "Global Geopolitics",
    "Artificial Intelligence",
    "science and technology",
    "international relations India",
    "Computer Science trending",
    "Generative AI",
]

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


# ── Security & Session Helpers ─────────────────────────────────────
def hash_password(password, salt):
    return hashlib.sha256((password + salt).encode()).hexdigest()

def generate_token():
    return os.urandom(24).hex()

def get_user(username):
    try:
        resp = table.get_item(Key={"query_hash": f"USER#{username.lower()}"})
        return resp.get("Item")
    except Exception as e:
        print(f"Error fetching user: {e}")
        return None

def save_user(user_item):
    try:
        table.put_item(Item=user_item)
        return True
    except Exception as e:
        print(f"Error saving user: {e}")
        return False

def verify_session(username, token):
    if not username or not token:
        return False, None
    try:
        user_item = get_user(username)
        if user_item and user_item.get("session_token") == token:
            expires = float(user_item.get("session_expires", 0))
            if expires > time.time():
                return True, user_item
    except Exception as e:
        print(f"Session verification failed: {e}")
    return False, None


# ── Personalized News Briefing ─────────────────────────────────────
def handle_personalized_news(user_item):
    username = user_item["username"]
    preferred_topics = user_item.get("preferred_topics", [])
    if not preferred_topics:
        return {
            "statusCode": 200,
            "headers": cors_headers(),
            "body": json.dumps({
                "answer": "You haven't selected any preferred topics yet! Go to your profile settings to configure them.",
                "sources": [],
                "articles": [],
                "cached": False
            })
        }

    # Format date string for daily caching
    today = time.strftime("%Y-%m-%d", time.gmtime())
    cache_key = f"BRIEFING#{username.lower()}#{today}"
    
    # Check cache first
    try:
        resp = table.get_item(Key={"query_hash": cache_key})
        cached_item = resp.get("Item")
        if cached_item:
            print("Personalized briefing cache hit!")
            return {
                "statusCode": 200,
                "headers": cors_headers(),
                "body": json.dumps({
                    "answer": cached_item.get("briefing"),
                    "sources": cached_item.get("sources", []),
                    "articles": cached_item.get("articles", []),
                    "cached": True
                })
            }
    except Exception as e:
        print(f"Failed checking briefing cache: {e}")

    # Load FAISS index/metadata (skipped if warm start)
    try:
        load_index()
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": cors_headers(),
            "body": json.dumps({"error": f"Failed to load news database: {str(e)}"})
        }

    if _metadata is None:
        return {
            "statusCode": 500,
            "headers": cors_headers(),
            "body": json.dumps({"error": "News database is empty."})
        }

    # Filter chunks matching user's preferred topics
    matched_chunks = []
    topic_counts = {topic: 0 for topic in preferred_topics}
    
    for chunk in _metadata:
        # Get topic with fallback keyword matching
        topic = chunk.get("topic")
        if not topic:
            text = (chunk.get("title", "") + " " + chunk.get("chunk_text", "")).lower()
            for t in TOPICS:
                if t.lower() in text:
                    topic = t
                    break
        
        if topic in preferred_topics:
            # Represent each topic fairly by capping chunks
            if topic_counts[topic] < 3:
                matched_chunks.append(chunk)
                topic_counts[topic] += 1
                
    if not matched_chunks:
        return {
            "statusCode": 200,
            "headers": cors_headers(),
            "body": json.dumps({
                "answer": "No news updates found today matching your preferred topics. Check back tomorrow!",
                "sources": [],
                "articles": [],
                "cached": False
            })
        }

    # Extract unique articles
    seen_urls = set()
    articles = []
    for chunk in matched_chunks:
        url = chunk.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            articles.append({
                "title": chunk.get("title", "News Update"),
                "url": url,
                "date": chunk.get("date", ""),
                "source": chunk.get("source", ""),
                "topic": chunk.get("topic", "") or "General"
            })

    # Limit to top 10 chunks for briefing text context
    context_chunks = matched_chunks[:10]
    context_str = ""
    for i, chunk in enumerate(context_chunks):
        context_str += (
            f"[{i+1}] {chunk['chunk_text']}\n"
            f"    Source: {chunk['title']} — {chunk['url']}\n\n"
        )

    # Prompt Bedrock Nova Lite
    topics_list = ", ".join(preferred_topics)
    prompt = (
        f"You are a personalized daily news assistant. The user is interested in the following topics: {topics_list}.\n\n"
        f"Using ONLY the context below, generate a comprehensive daily news briefing covering these topics. "
        f"Group the briefing by topic with bold headings, and use bullet points for key news items. "
        f"Cite sources using [1], [2] etc. inline. End with a Sources section listing the unique URL for each citation.\n\n"
        f"Context:\n{context_str}"
    )

    try:
        briefing = call_nova(prompt)
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": cors_headers(),
            "body": json.dumps({"error": f"Failed to generate briefing using AI: {str(e)}"})
        }

    sources = extract_sources(context_chunks)

    # Cache briefing in DynamoDB (expires in 24 hours)
    try:
        table.put_item(Item={
            "query_hash": cache_key,
            "type": "personalized_briefing",
            "briefing": briefing,
            "sources": sources,
            "articles": articles,
            "expires_at": int(time.time()) + 86400
        })
    except Exception as e:
        print(f"Failed to cache personalized briefing: {e}")

    return {
        "statusCode": 200,
        "headers": cors_headers(),
        "body": json.dumps({
            "answer": briefing,
            "sources": sources,
            "articles": articles,
            "cached": False
        })
    }


# ── main handler ──────────────────────────────────────────────────
def handler(event, context):
    try:
        body  = json.loads(event.get("body") or "{}")
        action = body.get("action", "").strip().lower()
    except Exception:
        return {
            "statusCode": 400,
            "headers":    cors_headers(),
            "body":       json.dumps({"error": "Invalid request body"}),
        }

    # Default action is query if not specified (maintains backwards compatibility)
    if not action:
        query = body.get("query", "").strip()
        if not query:
            return {
                "statusCode": 400,
                "headers":    cors_headers(),
                "body":       json.dumps({"error": "query field or action is required"}),
            }
        action = "query"

    # -- Dispatch endpoints
    if action == "register":
        username = body.get("username", "").strip()
        password = body.get("password", "").strip()
        preferences = body.get("preferences", [])
        
        if not username or not password:
            return {
                "statusCode": 400,
                "headers": cors_headers(),
                "body": json.dumps({"error": "Username and password are required."})
            }
            
        if get_user(username):
            return {
                "statusCode": 400,
                "headers": cors_headers(),
                "body": json.dumps({"error": "Username is already taken."})
            }
            
        salt = os.urandom(16).hex()
        pwd_hash = hash_password(password, salt)
        
        user_item = {
            "query_hash": f"USER#{username.lower()}",
            "type": "user_profile",
            "username": username,
            "password_hash": pwd_hash,
            "salt": salt,
            "preferred_topics": preferences,
            "queries_used": 0,
            "registered_at": int(time.time())
        }
        
        if save_user(user_item):
            return {
                "statusCode": 200,
                "headers": cors_headers(),
                "body": json.dumps({"success": True, "message": "Registration successful!"})
            }
        else:
            return {
                "statusCode": 500,
                "headers": cors_headers(),
                "body": json.dumps({"error": "Failed to save user profile."})
            }

    elif action == "login":
        username = body.get("username", "").strip()
        password = body.get("password", "").strip()
        
        if not username or not password:
            return {
                "statusCode": 400,
                "headers": cors_headers(),
                "body": json.dumps({"error": "Username and password are required."})
            }
            
        user_item = get_user(username)
        if not user_item:
            return {
                "statusCode": 401,
                "headers": cors_headers(),
                "body": json.dumps({"error": "Invalid username or password."})
            }
            
        salt = user_item.get("salt")
        stored_hash = user_item.get("password_hash")
        
        if hash_password(password, salt) != stored_hash:
            return {
                "statusCode": 401,
                "headers": cors_headers(),
                "body": json.dumps({"error": "Invalid username or password."})
            }
            
        token = generate_token()
        expires = int(time.time()) + 86400 * 7  # 7 days
        
        user_item["session_token"] = token
        user_item["session_expires"] = expires
        
        if save_user(user_item):
            return {
                "statusCode": 200,
                "headers": cors_headers(),
                "body": json.dumps({
                    "success": True,
                    "username": user_item.get("username"),
                    "token": token,
                    "preferences": user_item.get("preferred_topics", []),
                    "queries_used": int(user_item.get("queries_used", 0))
                })
            }
        else:
            return {
                "statusCode": 500,
                "headers": cors_headers(),
                "body": json.dumps({"error": "Failed to store session token."})
            }

    elif action == "update_preferences":
        username = body.get("username", "").strip()
        token = body.get("token", "").strip()
        preferences = body.get("preferences", [])
        
        is_valid, user_item = verify_session(username, token)
        if not is_valid:
            return {
                "statusCode": 401,
                "headers": cors_headers(),
                "body": json.dumps({"error": "Invalid or expired session."})
            }
            
        user_item["preferred_topics"] = preferences
        
        # Clear cached daily briefing for today to force regeneration
        today = time.strftime("%Y-%m-%d", time.gmtime())
        cache_key = f"BRIEFING#{username.lower()}#{today}"
        try:
            table.delete_item(Key={"query_hash": cache_key})
        except Exception:
            pass
            
        if save_user(user_item):
            return {
                "statusCode": 200,
                "headers": cors_headers(),
                "body": json.dumps({"success": True, "preferences": preferences})
            }
        else:
            return {
                "statusCode": 500,
                "headers": cors_headers(),
                "body": json.dumps({"error": "Failed to update preferences."})
            }

    elif action == "get_personalized_news":
        username = body.get("username", "").strip()
        token = body.get("token", "").strip()
        
        is_valid, user_item = verify_session(username, token)
        if not is_valid:
            return {
                "statusCode": 401,
                "headers": cors_headers(),
                "body": json.dumps({"error": "Invalid or expired session."})
            }
            
        return handle_personalized_news(user_item)

    elif action == "query":
        query = body.get("query", "").strip()
        mode  = body.get("mode", "answer").strip().lower()
        username = body.get("username", "").strip()
        token = body.get("token", "").strip()
        
        if not query:
            return {
                "statusCode": 400,
                "headers":    cors_headers(),
                "body":       json.dumps({"error": "query field is required"}),
            }
            
        queries_used = 0
        is_admin = False
        
        # Apply rate limiting if requested by a logged-in user context
        if username and token:
            is_valid, user_item = verify_session(username, token)
            if not is_valid:
                return {
                    "statusCode": 401,
                    "headers": cors_headers(),
                    "body": json.dumps({"error": "Invalid or expired session."})
                }
            
            is_admin = (username.lower() == "admin")
            queries_used = int(user_item.get("queries_used", 0))
            
            if not is_admin and queries_used >= 5:
                return {
                    "statusCode": 403,
                    "headers": cors_headers(),
                    "body": json.dumps({"error": "Query limit reached. Demo accounts are limited to 5 query chances."})
                }
            
            # Increment counter
            if not is_admin:
                try:
                    resp = table.update_item(
                        Key={"query_hash": f"USER#{username.lower()}"},
                        UpdateExpression="SET queries_used = queries_used + :val",
                        ExpressionAttributeValues={":val": 1},
                        ReturnValues="UPDATED_NEW"
                    )
                    queries_used = int(resp["Attributes"]["queries_used"])
                except Exception as e:
                    print(f"Failed atomic counter update: {e}")
                    user_item["queries_used"] = queries_used + 1
                    save_user(user_item)
                    queries_used = user_item["queries_used"]

        # -- load faiss index (skipped on warm starts)
        load_index()

        # -- check cache first
        cache_key = hashlib.md5(f"{query}:{mode}".encode()).hexdigest()
        cached    = check_cache(cache_key)
        if cached:
            # Inject current queries_used count in cached responses
            try:
                cached_dict = json.loads(cached)
                cached_dict["queries_used"] = queries_used
                cached = json.dumps(cached_dict)
            except Exception:
                pass
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
            "queries_used": queries_used
        })

        # -- write to cache
        write_cache(cache_key, result)

        return {
            "statusCode": 200,
            "headers":    cors_headers(),
            "body":       result,
        }
        
    else:
        return {
            "statusCode": 400,
            "headers": cors_headers(),
            "body": json.dumps({"error": f"Unknown action: {action}"})
        }


def cors_headers():
    return {
        "Content-Type":                "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }