# GCP infrastructure — `infra/gcp/`

Terraform for the GCP side of forecast-ml-framework (Phase 2).

## What it creates

| Resource | Purpose | Conf mapping |
|---|---|---|
| `google_storage_bucket.artifacts` (versioning on) | Pipeline artifacts bucket | `gs://forecast-ml-artifacts/...` in `conf/gcp/catalog.yml` |
| `google_storage_bucket.temp` (versioning on, 7-day expiry) | Dataproc / Spark temp | `params:tmp_bucket` |
| `google_bigquery_dataset.forecast` | Staging + metrics tables | `params:data_project` / `params:db_staging_data` |
| `google_dataproc_cluster.dev` (single-node, `count = var.enable_dataproc ? 1 : 0`) | Dev Spark cluster | Spark jobs |
| `google_dataproc_autoscaling_policy.scaling` (`count = var.autoscaling_enabled && var.enable_dataproc ? 1 : 0`, default **false**) | Dataproc autoscaling policy (workers scale min→max) | Spark jobs at 100M rows, when enabled |
| `google_cloud_scheduler_job.retraining` (`count = var.enable_retraining_scheduler ? 1 : 0`, default **false**) | Retraining scheduler (cron `retraining_schedule_cron`, default weekly `"0 6 * * 1"`) | Triggers retraining runs per `retraining_params.schedule_days` |
| `google_pubsub_topic.alerts` | Pub/Sub topic for monitoring alerts (catalog: `alerts`) | Alert sink; email subscription is a commented stub |
| `google_pubsub_subscription.alerts_email` (**commented stub**) | Email subscription stub for alerts | Uncomment + set `alerts_email` and a real email-bridge endpoint to provision |

Defaults match `conf/` (`artifacts_bucket = "forecast-ml-artifacts"`,
`bigquery_dataset_id = "forecast_dataset"`, `temp_bucket = "forecast-ml-temp"`).
`project_id` has **no default** — it is required and must be supplied by the
caller (variable only, never hardcoded).

## Prerequisites

- Terraform `>= 1.3` (`terraform version`).
- GCP credentials from the environment only — set
  `GOOGLE_APPLICATION_CREDENTIALS` (path to a service-account key) or use
  Application Default Credentials (`gcloud auth application-default login`).
  Never put credentials in `.tf`, `.tfvars`, or `conf/`.

## Apply order

Run from this directory:

```bash
terraform init
terraform plan -var="project_id=YOUR_PROJECT_ID"
terraform apply -var="project_id=YOUR_PROJECT_ID"
```

Full lifecycle (after reviewing the plan):

```bash
terraform plan -var="project_id=YOUR_PROJECT_ID"
terraform apply -var="project_id=YOUR_PROJECT_ID"
terraform destroy -var="project_id=YOUR_PROJECT_ID"
```

Or export `TF_VAR_project_id=YOUR_PROJECT_ID` and omit `-var`.
Other variables are optional (`terraform plan -var="region=europe-west1" ...`).

## State

Local state by default (`terraform.tfstate`, gitignored — do not commit it).
For shared use, adopt the shipped `gcp/backend.tf.example` (GCS, A6)
by copying it to `backend.tf`; the pattern is:

```hcl
# terraform {
#   backend "gcs" {
#     bucket = "forecast-ml-tfstate"  # create this bucket first, out of band
#     prefix = "forecast-ml/gcp"
#   }
# }
```

## Outputs → conf overlays

After apply, `terraform output` gives values to wire into the Kedro env overlay:

- `artifacts_bucket_url` (`gs://forecast-ml-artifacts`) — already the default
  in `conf/gcp/catalog.yml`; if you override `artifacts_bucket`, set
  the matching `params:artifacts_bucket` runtime param
  (`conf/base/parameters/platform.yml:51-61`, see `conf/README.md`).
- `temp_bucket_name` — set `params:tmp_bucket` in your run parameters.
- `bigquery_dataset_id` — set `params:db_staging_data` (and
  `params:data_project` to the same project).
- `dataproc_cluster_name` — identifies the dev cluster for Spark jobs.

## Encryption (CMEK) + workload IAM scope

- **CMEK — production MUST set `kms_key_name`** (R-06):
  `kms_key_name = ""` (default) uses Google-managed encryption
  (dev only). Production MUST set it to a real KMS key path
  (`infra/gcp/variables.tf:121-130`; applied at `infra/gcp/main.tf:78-83,123-128`).
  Validation enforces it (D7/X7): `production = true` with an empty
  `kms_key_name` fails `terraform plan`.
- **Workload IAM scope (R-08):** the dedicated workload SA
  (`forecast-ml-workloads`) gets `roles/storage.objectViewer` on the
  artifacts bucket (read-only — it never writes artifacts directly)
  and `roles/storage.objectAdmin` only on the separate temp bucket
  for Dataproc/Spark staging writes (`infra/gcp/main.tf:39-56`).
  Do not widen the artifacts binding back to `objectAdmin`; a job
  that must write artifacts needs a separate, narrow, reviewed
  write binding.

## Costs

- GCS: storage + requests for the artifacts/temp buckets (artifacts
  noncurrent versions expire after 90 days, `with_state = "ARCHIVED"`,
  `infra/gcp/main.tf:94-102`; temp objects auto-delete after 7 days,
  `with_state = "ANY"`, `infra/gcp/main.tf:134-142`).
- BigQuery: dataset itself is free; storage/queries against tables cost.
- Dataproc: **off by default (`enable_dataproc = false`, count = 0)**
  so no billing until enabled; when enabled it is billed per
  cluster-hour while it exists — destroy it when not in use
  (`terraform destroy` or set `enable_dataproc = false` and re-apply).
  Buckets enforce `public_access_prevention = "enforced"` and the
  cluster sets `internal_ip_only = true`.

## 100M-row sizing / Dataproc autoscaling

Local pandas is capped by `pandas_conversion_params.max_rows_for_pandas`
(default 5000000); 100M-row scoring must use the Spark path
(`scoring_params.spark_scoring.enabled = true`,
`partition_batch_size = 10000`, `max_rows = null` = unbounded).

- **Enable for a cloud run:** `enable_dataproc = true` and
  `autoscaling_enabled = true`. The cluster's `autoscaling_config`
  references `google_dataproc_autoscaling_policy.scaling`, which
  scales workers between `autoscaling_min_workers` (default 0) and
  `autoscaling_max_workers` (default 10) under CPU oversubscription
  (`min_instances`/`max_instances`).
- **Worker tiers:** dev = single-node (0 workers); 100M rows =
  autoscaling up to `autoscaling_max_workers` (raise the max only
  for the run — it bounds cost).
- **Shuffle:** `spark_params` in `conf/base/parameters/platform.yml`
  (`platform.yml:19-24`) sets
  `spark.sql.shuffle.partitions = 1000` (~100k rows/partition at
  100M rows), `spark.dynamicAllocation.enabled = 'true'`
  (min 2 / max 20 *executors*, `conf/base/parameters/platform.yml:29-32`)
  and `spark.sql.adaptive.enabled = 'true'` (AQE coalesces small
  partitions). Dataproc autoscaling (`autoscaling_min/max_workers`,
  *workers*) is a separate knob — do not conflate the two.
- **BigQuery:** catalog tables are partitioned by `run_date` (DAY)
  and clustered on `score_model_id` — keep new tables the same so
  100M-row scans prune partitions.
- **Cost note:** `autoscaling_enabled` defaults to **false** — no
  policy and no worker billing until enabled. When enabled, workers
  are billed while the cluster runs; set `autoscaling_enabled =
  false` (or destroy) after the run.

```bash
terraform apply -var="project_id=YOUR_PROJECT_ID" \
  -var="enable_dataproc=true" -var="autoscaling_enabled=true" \
  -var="autoscaling_max_workers=20"
```

## Phase 3: retraining scheduler + alerts topic

- **Scheduler:** `google_cloud_scheduler_job.retraining` is
  count-gated (`count = var.enable_retraining_scheduler ? 1 : 0`) and
  the variable defaults to **false**, so no scheduled retraining runs
  by default. To enable: set `enable_retraining_scheduler = true`,
  `retraining_schedule_cron` (default `"0 6 * * 1"`, weekly — matches
  `retraining_params.schedule_days = 7`), and `retraining_job_uri` to
  the real retraining endpoint (Cloud Run / Cloud Build trigger);
  the default URI is a stub placeholder (`example.invalid`).
- **Alerts:** `google_pubsub_topic.alerts` receives monitoring alerts
  (catalog dataset `alerts`). The email subscription
  `google_pubsub_subscription.alerts_email` is a **commented stub** —
  uncomment it, set `alerts_email`, and point `push_endpoint` at a
  real email-bridge endpoint to provision; until then no email is sent.
- **Cost note:** the scheduler job and topic cost nothing while the
  scheduler is disabled / the topic is idle; when enabled, each firing
  triggers a retraining run billed per run.

```bash
terraform apply -var="project_id=YOUR_PROJECT_ID" \
  -var="enable_retraining_scheduler=true" \
  -var="retraining_job_uri=https://YOUR_RETRAINING_ENDPOINT"
```

## Validation

```bash
terraform fmt -check -recursive .
terraform validate
```
