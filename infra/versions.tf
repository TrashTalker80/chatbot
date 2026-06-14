terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment and populate for remote state (recommended before first apply):
  # backend "s3" {
  #   bucket         = "your-tfstate-bucket"
  #   key            = "appther-chatbot/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "your-tfstate-lock-table"
  #   encrypt        = true
  # }
}
