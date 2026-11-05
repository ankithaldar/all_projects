# GCP infrastructure for forecast-ml-framework (Phase 2).
#
# Creates: GCS artifacts bucket (versioning on), GCS temp bucket,
# BigQuery dataset, and a single-node Dataproc dev cluster.
# Auth comes from the environment only (GOOGLE_APPLICATION_CREDENTIALS /
# Application Default Credentials); nothing is hardcoded here.

terraform {
  required_version = ">= 1.3"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- Artifacts bucket ---------------------------------------------------------
# Holds everything conf/gcp/catalog.yml points at
# (gs://forecast-ml-artifacts/data/..., .../artifacts/...).

resource "google_storage_bucket" "artifacts" {
  name                        = var.artifacts_bucket
  project                     = var.project_id
  location                    = var.bucket_location
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  labels = var.labels
}

# --- Temporary bucket ---------------------------------------------------------
# Staging / Dataproc temp bucket (params:tmp_bucket). Objects expire
# after 7 days so dev runs do not accumulate cost.

resource "google_storage_bucket" "temp" {
  name                        = var.temp_bucket
  project                     = var.project_id
  location                    = var.bucket_location
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 7
    }
  }

  labels = var.labels
}

# --- BigQuery dataset ---------------------------------------------------------
# Staging + metrics tables (params:data_project / params:db_staging_data).

resource "google_bigquery_dataset" "forecast" {
  dataset_id                 = var.bigquery_dataset_id
  project                    = var.project_id
  location                   = var.bigquery_location
  description                = "Forecast ML staging and metrics dataset."
  delete_contents_on_destroy = false

  labels = var.labels
}

# --- Dataproc cluster (single-node dev config) --------------------------------
# Single-node: master_config num_instances = 1, worker_config
# num_instances = 0 (Dataproc single-node pattern). Variable-driven via
# var.enable_dataproc (count), var.dataproc_* and var.labels.

resource "google_dataproc_cluster" "dev" {
  count   = var.enable_dataproc ? 1 : 0
  name    = var.dataproc_cluster_name
  project = var.project_id
  region  = var.region

  cluster_config {
    gce_cluster_config {
      zone           = var.dataproc_zone
      metadata       = var.labels
      internal_ip_only = true
    }

    staging_bucket = google_storage_bucket.temp.name

    master_config {
      num_instances = 1
      machine_type  = var.dataproc_machine_type
    }

    # Single-node dev config: no workers.
    worker_config {
      num_instances = 0
    }

    software_config {
      image_version = var.dataproc_image_version == "" ? null : var.dataproc_image_version
    }

    # Autoscaling: attach the policy only when enabled (and the
    # cluster itself exists). Cloud-scale Spark (100M rows) scales
    # workers between var.autoscaling_min/max_workers on CPU.
    dynamic "autoscaling_config" {
      for_each = var.autoscaling_enabled && var.enable_dataproc ? [1] : []
      content {
        policy = google_dataproc_autoscaling_policy.scaling[0].name
      }
    }
  }

  labels = var.labels
}

# --- Dataproc autoscaling policy (100M-row cloud scaling) ---------------------
# Scale bounds: workers scale between var.autoscaling_min_workers
# and var.autoscaling_max_workers (min_instances/max_instances)
# under CPU oversubscription. Off by default
# (var.autoscaling_enabled = false, count = 0): no policy, no
# autoscaling billing. COST NOTE: when enabled, workers are billed
# while the cluster runs -- idle/destroy the cluster after a run.

resource "google_dataproc_autoscaling_policy" "scaling" {
  count    = var.autoscaling_enabled && var.enable_dataproc ? 1 : 0
  name     = var.autoscaling_policy_name
  project  = var.project_id
  location = var.region

  policy {
    worker_config {
      num_instances {
        min = var.autoscaling_min_workers
        max = var.autoscaling_max_workers
      }
    }

    basic_scaling_config {
      scaling_config {
        min_instances        = var.autoscaling_min_workers
        max_instances        = var.autoscaling_max_workers
        cpu_oversubscription = 2.0
      }
    }
  }
}
