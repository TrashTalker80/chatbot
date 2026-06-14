output "index_bucket_name" {
  value       = aws_s3_bucket.index.bucket
  description = "S3 bucket for the LanceDB vector index"
}

output "widget_bucket_name" {
  value       = aws_s3_bucket.widget_static.bucket
  description = "S3 bucket for chat widget static assets"
}

output "dynamodb_table_name" {
  value       = aws_dynamodb_table.main.name
  description = "DynamoDB table (answer cache, feedback, leads, content-gap log)"
}

output "ecr_crawler_url" {
  value       = aws_ecr_repository.crawler.repository_url
  description = "ECR repository URL for the crawler Docker image"
}

output "lambda_role_arn" {
  value       = aws_iam_role.lambda.arn
  description = "IAM role ARN for the Lambda function"
}

output "crawler_role_arn" {
  value       = aws_iam_role.crawler.arn
  description = "IAM role ARN for the crawler Fargate task"
}

output "voyage_secret_arn" {
  value       = aws_secretsmanager_secret.voyage_api_key.arn
  description = "Secrets Manager ARN for the Voyage API key"
}

output "gemini_secret_arn" {
  value       = aws_secretsmanager_secret.gemini_api_key.arn
  description = "Secrets Manager ARN for the Gemini API key"
}
