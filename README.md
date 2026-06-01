# NewsRAG — Serverless AI News Intelligence

> A near real-time current affairs Q&A and quiz engine built entirely on AWS serverless infrastructure. Ask questions about today's news, get cited answers, or auto-generate MCQ quizzes — all for under **$0.30/month**.

![Architecture](./newsrag_architecture.svg)

---

## Live Demo

**[NewsRAG](https://news-rag-alpha.vercel.app/)** — try asking *"What are the latest developments in AI?"*

---

## What it does

NewsRAG ingests live news articles daily, indexes them as semantic vectors, and answers student questions using retrieval-augmented generation. Two modes:

- **Answer mode** — returns a journalist-style briefing with inline citations and source links
- **Quiz mode** — generates 5 interactive MCQs from retrieved news chunks

---

## Architecture overview

The system is split into two decoupled pipelines that share an S3 store.

### Ingestion pipeline (runs daily at 6 AM IST)

```
EventBridge Scheduler
      │
      ▼
Lambda: rag-ingest
  ├── Fetches NewsAPI top headlines (5 topic queries × 10 articles)
  ├── Chunks each article into ~300-character pieces
  ├── Embeds each chunk via Bedrock Titan Text Embeddings V2
  ├── Builds a FAISS IndexFlatL2 from all vectors
  ├── Saves  index/faiss.index     → S3
  ├── Saves  index/metadata.json   → S3
  └── Saves  raw/YYYY-MM-DD.json   → S3 (data lake)
```

### Query pipeline (on-demand, per user request)

```
Student (browser)
      │  POST /query  { query, mode }
      ▼
API Gateway (REST, prod stage)
      │
      ▼
Lambda: rag-query
  ├── Cold start: downloads faiss.index + metadata.json from S3 → /tmp
  ├── Warm start: reuses globals (no S3 download)
  ├── Embeds query via Bedrock Titan
  ├── Runs FAISS kNN search → top 3 chunks
  ├── Checks DynamoDB cache (24h TTL)
  │     ├── Cache HIT  → return immediately
  │     └── Cache MISS → call Bedrock Nova Lite
  │           ├── Answer mode: journalist-style briefing with citations
  │           └── Quiz mode:   5 MCQs with 4 options + answer key
  ├── Writes result to DynamoDB cache
  └── Returns { answer, sources, mode, cached }
      │
      ▼
CloudFront → S3 static frontend
```

---

## AWS services used

| Service | Role | Why |
|---|---|---|
| **EventBridge Scheduler** | Triggers daily ingestion | Zero cost, cron-based |
| **Lambda** (×2) | Ingestion + query compute | Pay per invocation, no idle cost |
| **S3** | FAISS index store + raw data lake | ~$0.01/mo at this scale |
| **Bedrock Titan Embed V2** | Text → vector embeddings | 1024-dim, Amazon-native |
| **Bedrock Nova Lite** | RAG answer + quiz generation | Cheapest capable Amazon model |
| **FAISS** (Lambda layer) | Vector similarity search | In-memory, no DB cost |
| **DynamoDB** | 24h response cache | On-demand, free tier eligible |
| **API Gateway** | REST endpoint + throttling | Rate-limited to 5 req/sec |
| **Secrets Manager** | NewsAPI key storage | Never hardcoded |
| **IAM** | Least-privilege role | Single role for both Lambdas |
| **CloudWatch** | Logs + cost alerts | Observability |

---

## Cost breakdown

Measured at portfolio/demo traffic (~50 queries/month):

| Service | Monthly cost |
|---|---|
| Lambda (both functions) | $0.00 (free tier) |
| S3 storage + requests | ~$0.01 |
| Bedrock Titan Embeddings | ~$0.01 |
| Bedrock Nova Lite | ~$0.05–0.15 |
| DynamoDB (on-demand) | $0.00 (free tier) |
| EventBridge | $0.00 (free tier) |
| API Gateway | $0.00 (free tier) |
| **Total** | **~$0.07–0.17/mo** |

The key cost decisions: FAISS-in-S3 instead of OpenSearch Serverless (saves ~$75/mo), EventBridge batch ingestion instead of Kinesis streaming, and DynamoDB caching to avoid duplicate Bedrock calls.

---

## Tech stack

**Python 3.11** · **boto3** · **faiss-cpu 1.7.4** · **numpy 1.26.4** · **AWS Lambda** · **Amazon Bedrock**

---

## Project structure

```
news-rag-aws/
├── ingest/
│   └── lambda_function.py     # Daily ingestion: fetch → chunk → embed → index
├── query/
│   └── lambda_function.py     # Query: embed → retrieve → generate → cache
├── frontend/
│   └── index.html             # Single-page UI (S3 + CloudFront)
├── layer/
│   └── (faiss-layer.zip)      # FAISS Lambda layer — built via Docker
└── README.md
```

---

## Local setup / deployment guide

### Prerequisites
- AWS account with Bedrock model access (Titan Embed V2, Nova Lite)
- Docker Desktop (for building the FAISS layer)
- NewsAPI free account

### Step 1 — Build the FAISS Lambda layer
```bash
# Windows CMD
docker run --rm --entrypoint pip -v %cd%:/out ^
  public.ecr.aws/lambda/python:3.11 ^
  install faiss-cpu==1.7.4 numpy==1.26.4 packaging -t /out/python

powershell -Command "Compress-Archive -Path python -DestinationPath faiss-layer.zip"
```

Upload `faiss-layer.zip` to **Lambda → Layers → Create layer** (Python 3.11).

### Step 2 — Create infrastructure (AWS Console)

| Resource | Config |
|---|---|
| S3 bucket | `your-rag-store` with `index/` and `raw/` folders |
| DynamoDB table | `rag-response-cache`, PK: `query_hash`, TTL: `expires_at` |
| Secrets Manager | `your-project/newsapi-key` → `{"api_key": "..."}` |
| IAM role | `lambda-rag-role` with S3, DynamoDB, Bedrock, Secrets Manager, CloudWatch |

### Step 3 — Deploy Lambda functions

**rag-ingest**
- Runtime: Python 3.11, Memory: 512 MB, Timeout: 5 min
- Layer: faiss-layer
- Env vars: `S3_BUCKET`, `SECRET_NAME`
- Trigger: EventBridge `cron(30 0 * * ? *)` (6 AM IST daily)

**rag-query**
- Runtime: Python 3.11, Memory: 1024 MB, Timeout: 30 sec
- Layer: faiss-layer
- Env vars: `S3_BUCKET`, `DYNAMODB_TABLE`
- Trigger: API Gateway POST `/query`

### Step 4 — Deploy frontend

Deploy the frontend in Vercel free tier

---

## Key engineering decisions

**Why FAISS-in-S3 instead of a vector database?**
At portfolio scale, OpenSearch Serverless costs a minimum of ~$75/month due to its 0.5 OCU floor. FAISS serialized to S3 costs fractions of a cent. The tradeoff is a ~1s cold-start download penalty, which is imperceptible at demo traffic.

**Why decouple ingestion from querying?**
Rebuilding the vector index on every query would be expensive and slow (~15s). Daily batch ingestion amortises the embedding cost across all queries for that day. This is a standard data engineering pattern — ETL separated from serving.

**Why DynamoDB for caching?**
Repeat queries (especially from employers demoing the same questions) would otherwise each trigger a Bedrock call. A 24h TTL cache reduces Bedrock costs to near zero at demo scale and improves response latency from ~2.5s to ~300ms on warm cache hits.

**Why Amazon Nova Lite over Claude Haiku?**
Keeps the entire stack within Amazon's ecosystem, reduces cross-vendor dependency, and Nova Lite is marginally cheaper at this scale.

---

## What this demonstrates

- **Data Engineering**: batch ingestion pipeline, S3 data lake with date partitioning, schema-on-read with raw JSON archive
- **GenAI / RAG**: embedding-based semantic retrieval, prompt engineering for news agent persona, cited answer generation
- **Serverless architecture**: event-driven compute, stateless Lambdas with warm-start optimisation, on-demand pricing
- **Cost engineering**: conscious tradeoffs between managed services and DIY to hit $0.30/mo target
- **AWS proficiency**: 10+ services wired together with IAM, Secrets Manager, and CloudWatch

---

## Vibe coding alert:
The frontend of this project was developed by AI.
