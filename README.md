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
pytest -q   # 262 tests across crawler (Steps 1–3) and api
```

## Implementation steps

See `plans/` and the full architecture doc for the 10-step build plan.
