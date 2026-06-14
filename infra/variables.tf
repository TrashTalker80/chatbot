variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name prefix for all resource names"
  type        = string
  default     = "appther-chatbot"
}

variable "bucket_suffix" {
  description = "Globally-unique suffix for S3 bucket names (use your 12-digit AWS account ID)"
  type        = string
}

variable "budget_alert_email" {
  description = "Email address that receives AWS Budget cost alerts"
  type        = string
}

variable "secret_recovery_window_days" {
  description = "Recovery window for Secrets Manager secrets (0 in dev, 7-30 in prod)"
  type        = number
  default     = 7
}
