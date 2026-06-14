# ── Lambda execution role ──────────────────────────────────────────────────────

resource "aws_iam_role" "lambda" {
  name = "${var.project}-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_app" {
  name = "${var.project}-lambda-app"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3IndexRead"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.index.arn,
          "${aws_s3_bucket.index.arn}/*",
        ]
      },
      {
        Sid    = "DynamoDB"
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
          "dynamodb:BatchWriteItem",
        ]
        Resource = [
          aws_dynamodb_table.main.arn,
          "${aws_dynamodb_table.main.arn}/index/*",
        ]
      },
      {
        Sid    = "SecretsRead"
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        Resource = [
          aws_secretsmanager_secret.voyage_api_key.arn,
          aws_secretsmanager_secret.gemini_api_key.arn,
          aws_secretsmanager_secret.jina_api_key.arn,
        ]
      },
    ]
  })
}

# ── Crawler task role (Fargate / GitHub Actions OIDC) ─────────────────────────

resource "aws_iam_role" "crawler" {
  name = "${var.project}-crawler"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Fargate task principal
        Effect    = "Allow"
        Principal = { Service = "ecs-tasks.amazonaws.com" }
        Action    = "sts:AssumeRole"
      },
      # GitHub Actions OIDC — uncomment after adding the OIDC provider to your account:
      # {
      #   Effect = "Allow"
      #   Principal = {
      #     Federated = "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      #   }
      #   Action = "sts:AssumeRoleWithWebIdentity"
      #   Condition = {
      #     StringEquals = {
      #       "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
      #       "token.actions.githubusercontent.com:sub" = "repo:YOUR_ORG/appther-chatbot:ref:refs/heads/main"
      #     }
      #   }
      # }
    ]
  })
}

resource "aws_iam_role_policy" "crawler_app" {
  name = "${var.project}-crawler-app"
  role = aws_iam_role.crawler.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3IndexWrite"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.index.arn,
          "${aws_s3_bucket.index.arn}/*",
        ]
      },
      {
        Sid    = "SecretsRead"
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        Resource = [
          aws_secretsmanager_secret.voyage_api_key.arn,
          aws_secretsmanager_secret.jina_api_key.arn,
        ]
      },
    ]
  })
}
