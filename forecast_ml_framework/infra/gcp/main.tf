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
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- Dedicated workload service account (MLops-13, Antisec F-07) -------------
# Dataproc nodes and the retraining scheduler OIDC identity both run as
# this dedicated least-privilege service account -- never the project
# default compute service account (historically broad scopes). The
# account carries no project-wide role; its only grants are the
# bucket-scoped object bindings below. Do not add broader bindings
# without a separate IAM review.

resource "google_service_account" "workloads" {
  project      = var.project_id
  account_id   = "forecast-ml-workloads"
  display_name = "forecast-ml-framework workloads (Dataproc + Scheduler OIDC)"
  description  = "Dedicated least-privilege identity for the Dataproc cluster and the retraining scheduler OIDC token. Replaces the default compute SA (MLops-13/F-07)."
}

# Least-privilege bucket access (Antisec R-08): the workload SA
# reads pipeline artifacts but never writes them directly. Dataproc /
# Spark staging and temp writes go to the separate temp bucket below
# (staging_bucket = temp), which keeps objectAdmin. If a future job
# must write artifacts directly, grant it a separate, narrow write
# binding in a reviewed change — do not widen this binding back to
# objectAdmin.
resource "google_storage_bucket_iam_member" "workloads_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.workloads.email}"
}

resource "google_storage_bucket_iam_member" "workloads_temp" {
  bucket = google_storage_bucket.temp.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.workloads.email}"
}

# --- Artifacts bucket ---------------------------------------------------------
# Holds everything conf/gcp/catalog.yml points at
# (gs://forecast-ml-artifacts/data/..., .../artifacts/...).

resource "google_storage_bucket" "artifacts" {
  name                        = var.artifacts_bucket
  project                     = var.project_id
  location                    = var.bucket_location
  uniform_bucket_level_access = true
  # Deny-public (F-12): PAP enforced + UBLA + no allUsers bindings
  # anywhere in this stack. Never add a public IAM binding here.
  public_access_prevention = "enforced"

  versioning {
    enabled = true
  }

  # CMEK (F-12): customer-managed key when var.kms_key_name is set;
  # Google-managed encryption otherwise (dev default -- production
  # must set kms_key_name, see variables.tf).
  dynamic "encryption" {
    for_each = var.kms_key_name != "" ? [1] : []
    content {
      default_kms_key_name = var.kms_key_name
    }
  }

  # Lifecycle (MLops R4): versioning is on, so without this rule
  # noncurrent versions of artifacts accumulate and are billed
  # forever -- the same gap cycle-1 flagged for AWS. Expire
  # noncurrent versions after 90 days, matching the AWS artifacts
  # bucket noncurrent_version_expiration (90d). Live (current)
  # objects are never expired by this rule: with_state = "ARCHIVED"
  # (GCS name for noncurrent versions) scopes it to old versions
  # only (unlike the temp bucket's ANY, which purges current +
  # noncurrent after 7 days).
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age        = 90
      with_state = "ARCHIVED"
    }
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

  # CMEK (F-12): same optional customer-managed key as artifacts.
  dynamic "encryption" {
    for_each = var.kms_key_name != "" ? [1] : []
    content {
      default_kms_key_name = var.kms_key_name
    }
  }

  # with_state = "ANY" is load-bearing (MLops-14): without it the
  # rule only matches LIVE objects, so with versioning on the
  # noncurrent versions would accumulate forever and the 7-day
  # expiry would not bound cost. ANY purges current + noncurrent.
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age        = 7
      with_state = "ANY"
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
      zone = var.dataproc_zone
      # Dedicated least-privilege SA (MLops-13/F-07) -- never the
      # default compute SA. Explicit network/subnetwork (required
      # when enabled, validated in variables.tf) -- never the
      # implicit default VPC.
      service_account  = google_service_account.workloads.email
      network          = var.dataproc_network
      subnetwork       = var.dataproc_subnetwork
      internal_ip_only = true
      # Scoped metadata (MLops-13): deliberately NOT var.labels --
      # resource labels are not VM metadata, and publishing every
      # label as VM metadata leaks internal tagging to any process
      # on the node. One fixed purpose key only.
      metadata = {
        "cluster-purpose" = "forecast-ml-dataproc"
      }
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
        policy_uri = google_dataproc_autoscaling_policy.scaling[0].id
      }
    }
  }

  labels = var.labels
}

# --- Dataproc autoscaling policy (100M-row cloud scaling) ---------------------
# Scale bounds: workers scale between var.autoscaling_min_workers
# and var.autoscaling_max_workers (min_instances/max_instances in
# worker_config). Off by default
# (var.autoscaling_enabled = false, count = 0): no policy, no
# autoscaling billing. COST NOTE: when enabled, workers are billed
# while the cluster runs -- idle/destroy the cluster after a run.

# --- Retraining scheduler (Phase 3, disabled by default) ----------------------
# count = var.enable_retraining_scheduler ? 1 : 0, and the variable
# defaults to false, so no scheduler job exists (and no retraining
# runs are triggered) until explicitly enabled. The cron comes from
# var.retraining_schedule_cron (default weekly "0 6 * * 1", matching
# retraining_params.schedule_days = 7). The job body is a stub: set
# var.retraining_job_uri to the real retraining endpoint (Cloud Run /
# Cloud Build trigger) before enabling. Least privilege: the job
# authenticates with OIDC as the dedicated workload service account
# (google_service_account.workloads), never the default compute SA;
# the target must verify the OIDC token (see http_target below).

resource "google_cloud_scheduler_job" "retraining" {
  count       = var.enable_retraining_scheduler ? 1 : 0
  name        = "forecast-ml-retraining"
  project     = var.project_id
  region      = var.region
  schedule    = var.retraining_schedule_cron
  time_zone   = "UTC"
  description = "Phase 3 retraining scheduler (stub body; set retraining_job_uri)."

  http_target {
    uri         = var.retraining_job_uri == "" ? "https://example.invalid/retraining-stub" : var.retraining_job_uri
    http_method = "POST"
    # OIDC auth (F-07): the scheduler POSTs with a short-lived OIDC
    # token for the dedicated workload SA, audience = the target
    # URI. The retraining target MUST verify this token (issuer
    # https://accounts.google.com, audience = its own URI); an
    # unauthenticated POST is never accepted. (The example.invalid
    # fallback is unreachable when the job is enabled -- the
    # validation on retraining_job_uri requires a real URI.)
    oidc_token {
      service_account_email = google_service_account.workloads.email
      audience              = var.retraining_job_uri == "" ? "https://example.invalid/retraining-stub" : var.retraining_job_uri
    }
  }

  # NOTE: google_cloud_scheduler_job has no labels argument in the
  # google provider ~> 5.0 schema -- intentionally omitted.
}

# --- Alerts Pub/Sub topic (Phase 3) -------------------------------------------
# Monitoring pipelines publish alerts (catalog: alerts) to this topic.
# The email subscription below is a commented stub: to provision it,
# uncomment and set var.alerts_email to a real address. Pub/Sub push
# subscriptions need a verified endpoint; email delivery requires a
# Pub/Sub -> email bridge (e.g. Cloud Functions / SendGrid), so the
# stub deliberately does nothing until that bridge exists.

resource "google_pubsub_topic" "alerts" {
  name    = var.alerts_topic_name
  project = var.project_id
  labels  = var.labels
}

# resource "google_pubsub_subscription" "alerts_email" {
#   name  = "forecast-ml-alerts-email"
#   topic = google_pubsub_topic.alerts.name
#
#   push_config {
#     push_endpoint = "https://example.invalid/alerts-email-bridge"  # set to a real email-bridge endpoint
#   }
#
#   ack_deadline_seconds = 20
# }

resource "google_dataproc_autoscaling_policy" "scaling" {
  count     = var.autoscaling_enabled && var.enable_dataproc ? 1 : 0
  policy_id = var.autoscaling_policy_name
  project   = var.project_id
  location  = var.region

  # google provider ~> 5.0 schema: flat worker_config with
  # min_instances/max_instances (no policy wrapper, no
  # basic_scaling_config; the old cpu_oversubscription knob is not
  # exposed by this schema version).
  worker_config {
    min_instances = var.autoscaling_min_workers
    max_instances = var.autoscaling_max_workers
  }

  basic_algorithm {
    cooldown_period = "300s"

    yarn_config {
      graceful_decommission_timeout  = "120s"
      scale_up_factor                = 2.0
      scale_down_factor              = 0.5
      scale_up_min_worker_fraction   = 0.0
      scale_down_min_worker_fraction = 0.0
    }
  }
}
