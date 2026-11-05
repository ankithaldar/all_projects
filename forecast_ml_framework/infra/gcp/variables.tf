# GCP variables. No credentials or project IDs are hardcoded anywhere in
# this module: the project must be supplied by the caller (see README.md,
# "Secrets via env only"). Bucket defaults match conf/gcp/catalog.yml
# (gs://forecast-ml-artifacts/...).

variable "project_id" {
  description = "GCP project ID. Required — no default (pass via -var or TF_VAR_project_id)."
  type        = string
}

variable "region" {
  description = "Default GCP region for regional resources."
  type        = string
  default     = "us-central1"
}

variable "artifacts_bucket" {
  description = "Name of the GCS bucket for pipeline artifacts. Matches conf/gcp/catalog.yml (gs://forecast-ml-artifacts/...)."
  type        = string
  default     = "forecast-ml-artifacts"
}

variable "temp_bucket" {
  description = "Name of the GCS bucket for Dataproc / Spark temporary data (params:tmp_bucket)."
  type        = string
  default     = "forecast-ml-temp"
}

variable "bucket_location" {
  description = "Location for the GCS buckets."
  type        = string
  default     = "US"
}

variable "bigquery_dataset_id" {
  description = "BigQuery dataset ID for staging/metrics tables (params:db_staging_data)."
  type        = string
  default     = "forecast_dataset"
}

variable "bigquery_location" {
  description = "BigQuery dataset location."
  type        = string
  default     = "US"
}

variable "enable_dataproc" {
  description = "Create the single-node Dataproc dev cluster. Default false (count = 0) so no billing by default; set true only when needed."
  type        = bool
  default     = false
}

variable "dataproc_cluster_name" {
  description = "Name of the Dataproc cluster."
  type        = string
  default     = "forecast-ml-dataproc"
}

variable "dataproc_zone" {
  description = "Zone for the Dataproc cluster."
  type        = string
  default     = "us-central1-a"
}

variable "dataproc_machine_type" {
  description = "Machine type for the Dataproc master node (single-node dev config)."
  type        = string
  default     = "n1-standard-4"
}

variable "dataproc_image_version" {
  description = "Dataproc image version. Empty string uses the provider default."
  type        = string
  default     = ""
}

# --- Dataproc autoscaling (100M-row cloud scaling) ---------------------------
# Off by default: no autoscaling policy (and no surprise worker
# billing) until explicitly enabled. COST NOTE: enabling scales
# workers between min/max on CPU pressure and is billed per worker
# while the cluster runs -- size max_workers to the 100M-row job
# and destroy/idle the cluster when done (see README.md).

variable "autoscaling_enabled" {
  description = "Attach a Dataproc autoscaling policy to the cluster. Default false (no policy, no autoscaling cost); set true only for cloud-scale Spark runs."
  type        = bool
  default     = false
}

variable "autoscaling_policy_name" {
  description = "Name of the Dataproc autoscaling policy."
  type        = string
  default     = "forecast-ml-scaling"
}

variable "autoscaling_min_workers" {
  description = "Minimum worker instances the autoscaling policy may scale to."
  type        = number
  default     = 0
}

variable "autoscaling_max_workers" {
  description = "Maximum worker instances the autoscaling policy may scale to. Bounds cost; raise only for 100M-row runs."
  type        = number
  default     = 10
}

variable "labels" {
  description = "Labels applied to supported resources."
  type        = map(string)
  default = {
    framework = "forecast-ml-framework"
    managedby = "terraform"
    purpose   = "dev"
  }
}
