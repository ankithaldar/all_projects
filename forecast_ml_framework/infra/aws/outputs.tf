# AWS outputs. Feed these into conf overlays / runtime params:
# artifacts_bucket_url -> s3://forecast-ml-artifacts (conf/aws/catalog.yml)
# athena_database      -> Athena database name

output "artifacts_bucket_name" {
  description = "Name of the S3 artifacts bucket."
  value       = aws_s3_bucket.artifacts.id
}

output "artifacts_bucket_arn" {
  description = "ARN of the S3 artifacts bucket."
  value       = aws_s3_bucket.artifacts.arn
}

output "artifacts_bucket_url" {
  description = "URI of the S3 artifacts bucket (matches conf/aws/catalog.yml)."
  value       = "s3://${aws_s3_bucket.artifacts.id}"
}

output "athena_database" {
  description = "Athena database name."
  value       = aws_athena_database.forecast.name
}

output "emr_cluster_id" {
  description = "ID of the EMR stub cluster (empty when enable_emr is false)."
  value       = var.enable_emr ? aws_emr_cluster.stub[0].id : ""
}
