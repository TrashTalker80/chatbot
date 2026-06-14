# recovery_window_in_days = 0 allows immediate deletion during development.
# Set to 7 or 30 in production.

resource "aws_secretsmanager_secret" "voyage_api_key" {
  name                    = "${var.project}/voyage-api-key"
  description             = "Voyage AI API key — embeddings (voyage-3.5) and reranking (rerank-2.5)"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret" "gemini_api_key" {
  name                    = "${var.project}/gemini-api-key"
  description             = "Google Gemini API key — LLM inference (Flash-Lite primary, 3 Flash escalation)"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret" "jina_api_key" {
  name                    = "${var.project}/jina-api-key"
  description             = "Jina AI API key — fallback/standby embeddings (jina-embeddings-v3)"
  recovery_window_in_days = 0
}
