# GCP outputs. Feed these into conf overlays / runtime params:
# artifacts_bucket_url -> gs://forecast-ml-artifacts (conf/gcp/catalog.yml)
# temp_bucket_name     -> params:tmp_bucket
# bigquery_dataset_id  -> params:db_staging_data

output "artifacts_bucket_name" {
  description = "Name of the GCS artifacts bucket."
  value       = google_storage_bucket.artifacts.name
}

output "artifacts_bucket_url" {
  description = "URI of the GCS artifacts bucket (matches conf/gcp/catalog.yml)."
  value       = "gs://${google_storage_bucket.artifacts.name}"
}

output "temp_bucket_name" {
  description = "Name of the GCS temp bucket (params:tmp_bucket)."
  value       = google_storage_bucket.temp.name
}

output "temp_bucket_url" {
  description = "URI of the GCS temp bucket."
  value       = "gs://${google_storage_bucket.temp.name}"
}

output "bigquery_dataset_id" {
  description = "BigQuery dataset ID (params:db_staging_data)."
  value       = google_bigquery_dataset.forecast.dataset_id
}

output "bigquery_dataset" {
  description = "Fully-qualified BigQuery dataset (project.dataset)."
  value       = "${var.project_id}.${google_bigquery_dataset.forecast.dataset_id}"
}

output "dataproc_cluster_name" {
  description = "Name of the Dataproc cluster (empty when enable_dataproc is false)."
  value       = var.enable_dataproc ? google_dataproc_cluster.dev[0].name : ""
}

output "dataproc_cluster_id" {
  description = "ID of the Dataproc cluster (empty when enable_dataproc is false)."
  value       = var.enable_dataproc ? google_dataproc_cluster.dev[0].id : ""
}

output "workloads_service_account_email" {
  description = "Email of the dedicated workload service account (Dataproc nodes + scheduler OIDC identity)."
  value       = google_service_account.workloads.email
}
