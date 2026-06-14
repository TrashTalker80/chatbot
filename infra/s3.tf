# Vector index — LanceDB files live here; versioning enables index rollback.
resource "aws_s3_bucket" "index" {
  bucket = "${var.project}-index-${var.bucket_suffix}"
}

resource "aws_s3_bucket_versioning" "index" {
  bucket = aws_s3_bucket.index.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "index" {
  bucket                  = aws_s3_bucket.index.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Widget static assets — served via CloudFront (Step 8), not directly public.
resource "aws_s3_bucket" "widget_static" {
  bucket = "${var.project}-widget-${var.bucket_suffix}"
}

resource "aws_s3_bucket_public_access_block" "widget_static" {
  bucket                  = aws_s3_bucket.widget_static.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
