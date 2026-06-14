# Appther RAG Chatbot

24/7 AI support chatbot for appther.com — serverless-first RAG pipeline on AWS.

**Architecture:** weekly crawler → Voyage embeddings → LanceDB-on-S3 → Lambda (hybrid retrieve + rerank) → Gemini Flash-Lite → streaming React widget.  
**Cost:** ~$16–20/month post-free-tier, $0 idle.

## Repo layout

```
crawler/    ingestion pipeline (discovery, fetch, clean, chunk, embed, index)
api/        FastAPI RAG endpoint (rewrite, retrieve, rerank, generate)
widget/     embeddable React chat widget
infra/      Terraform (S3, DynamoDB, Lambda, CloudFront, WAF, ECR, Secrets)
eval/       golden Q&A set + RAGAS harness + jailbreak probes
plans/      implementation notes and ADRs
```

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.12 |
| Node | 20 |
| Terraform | ≥ 1.6 |
| Docker | any recent |
| AWS CLI | configured with `us-east-1` access |

## Provider API keys

Store real values via:
```bash
aws secretsmanager put-secret-value \
  --secret-id appther-chatbot/voyage-api-key \
  --secret-string '{"api_key":"<YOUR_KEY>"}'
```

| Secret path | Provider | Used for |
|---|---|---|
| `appther-chatbot/voyage-api-key` | [Voyage AI](https://www.voyageai.com) | Embeddings (`voyage-3.5`) + reranking (`rerank-2.5`) |
| `appther-chatbot/gemini-api-key` | [Google AI Studio](https://aistudio.google.com) | LLM inference (Flash-Lite + 3 Flash) |
| `appther-chatbot/jina-api-key` | [Jina AI](https://jina.ai) | Fallback/standby embeddings (`jina-embeddings-v3`) |

## Terraform (scaffold / local validation)

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars   # fill in bucket_suffix + email
terraform init -backend=false                  # CI validation (no AWS needed)
terraform validate
# terraform apply                              # requires AWS credentials
```

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[crawler,embed,dev]"
pre-commit install
pytest -q   # 315 tests across crawler (Steps 1–4) and api
```

## Running the crawl pipeline

Full pipeline (discovers URLs, fetches, embeds, updates index, reconciles):

```bash
# Requires VOYAGE_API_KEY, JINA_API_KEY, and a LanceDB-accessible S3 URI
export LANCE_INDEX_URI=s3://your-bucket/lance_index
python -m crawler.pipeline
```

Useful flags:

| Flag | Effect |
|---|---|
| `--dry-run` | Discovery only — no fetches, no I/O |
| `--skip-embed` | Fetch + verify without touching the index |
| `--targeted` | Re-fetch only URLs from the previous run's `failures.jsonl` |
| `--staging-dir DIR` | Override output directory (default: `staging/`) |
| `-v` | Enable DEBUG logging |

After each run a JSON report is written to `staging/reports/report_<timestamp>.json` and a `staging/failures.jsonl` lists any pages that failed.

## GitHub Actions cron

The weekly crawl (`0 2 * * 0` — Sundays 02:00 UTC) is defined in `.github/workflows/crawl.yml`. Set the following repository secrets before the first run:

| Secret | Purpose |
|---|---|
| `VOYAGE_API_KEY` | Voyage AI embeddings + rerank |
| `JINA_API_KEY` | Jina standby embeddings |
| `AWS_ACCESS_KEY_ID` | S3 read/write for LanceDB index |
| `AWS_SECRET_ACCESS_KEY` | S3 credentials |
| `LANCE_INDEX_URI` | LanceDB S3 URI (e.g. `s3://bucket/lance_index`) |

The workflow uploads the report JSON as a CI artifact (retained 30 days) and fails the job if a drop-alert fires.

## Implementation steps

See `plans/` and the full architecture doc for the 10-step build plan.
