# Infrastructure — `infra/`

Terraform (IaC) for forecast-ml-framework, Phase 2. Two independent stacks:

| Stack | Directory | Creates |
|---|---|---|
| GCP (primary) | [`gcp/`](gcp/README.md) | GCS artifacts bucket (versioning on), GCS temp bucket, BigQuery dataset, single-node Dataproc dev cluster, retraining Cloud Scheduler job (off by default), alerts Pub/Sub topic |
| AWS (config-ready) | [`aws/`](aws/README.md) | S3 artifacts bucket (versioning on), Athena database, EMR stub (off by default), retraining EventBridge rule stub (commented) |

**Secrets via env only.** No credentials, tokens, keys, project IDs, or
account IDs are hardcoded in any `.tf` file. GCP auth comes from
`GOOGLE_APPLICATION_CREDENTIALS` / Application Default Credentials; AWS auth
from `AWS_PROFILE` or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`. The GCP
project ID is a required variable with no default; the AWS region/account
come from variables + environment.

## Apply order

The two stacks are independent — apply whichever you need (GCP first, per
the Phase 2 plan). Within each stack:

```bash
cd infra/gcp   # or infra/aws
terraform init
terraform plan -var="project_id=YOUR_PROJECT_ID"   # GCP only; AWS needs no -var
terraform apply -var="project_id=YOUR_PROJECT_ID"
```

Always review `terraform plan` output before `terraform apply`.
Tear down with `terraform destroy` (same `-var`).

## State

Local state (`terraform.tfstate`) by default — keep it out of git. For shared
or production use, adopt the shipped `backend.tf.example` per stack
(GCS for GCP, S3 + DynamoDB for AWS; A6 — commented, ignored by
Terraform, no active `backend.tf`) by copying it to `backend.tf` and
running `terraform init -migrate-state`. Never commit
`*.tfstate*` or `*.tfvars` containing real values. **No Docker** (A5):
serving is dev-only uvicorn.

## Costs (summary)

- **GCS/S3 storage:** artifacts buckets hold data + models + metrics;
  versioning is on, so overwritten objects keep old versions (watch growth).
  The GCP temp bucket auto-deletes objects after 7 days.
- **BigQuery/Athena:** the datasets/databases are free; queries scan data
  and are billed per TB scanned.
- **Dataproc/EMR compute:** the main cost driver. GCP Dataproc is
  **off by default** (`enable_dataproc = false`, count = 0) — enable
  only when needed and destroy it when idle. AWS EMR is **off by
  default** (`enable_emr = false`, count = 0) and costs nothing
  until enabled.

## How env vars map to `conf/` overlays

Terraform outputs are the source of truth for where bytes live; the Kedro
env overlays in `conf/` are the source of truth for what the pipeline reads.
They must agree:

| Terraform output | `conf/` setting |
|---|---|
| GCP `artifacts_bucket_url` (`gs://forecast-ml-artifacts`) | `filepath` values in `conf/gcp/catalog.yml` |
| GCP `temp_bucket_name` | `params:tmp_bucket` (runtime param) |
| GCP `bigquery_dataset_id` | `params:db_staging_data` (with `params:data_project` = the project) |
| AWS `artifacts_bucket_url` (`s3://forecast-ml-artifacts`) | `filepath` values in `conf/aws/catalog.yml` |
| AWS `athena_database` | Athena database for staging/metrics tables |

Bucket-name defaults already match the catalog filepaths
(`forecast-ml-artifacts`), so a default apply needs no `conf/` edits. If you
override a bucket/dataset name in Terraform, update the corresponding
`conf/{gcp,aws}/catalog.yml` filepaths (or runtime params) to match, then run
Kedro with `--env gcp` / `--env aws` and credentials from the environment
(`GOOGLE_APPLICATION_CREDENTIALS` / `AWS_PROFILE`) — never from YAML.

## 100M-row sizing guidance (cloud scaling)

Scoring >500 features over ~100M rows never fits the local pandas
path. The guard is `pandas_conversion_params.max_rows_for_pandas`
(default 5000000) in `conf/base/parameters/data.yml`: above it,
conversion raises and directs to the Spark path
(`scoring_params.spark_scoring.enabled = true`, with
`partition_batch_size = 10000` rows per partition sub-batch, and
`scoring_params.max_rows = null` = unbounded on Spark).

| Tier | Workers | Shuffle | When |
|---|---|---|---|
| Local / dev | `local[*]`, no cluster | `spark.sql.shuffle.partitions = 1000` (AQE coalesces small shuffles) | Small samples only; pandas guard applies |
| GCP Dataproc | Single-node dev by default; for 100M rows set `enable_dataproc = true` + `autoscaling_enabled = true` (`autoscaling_min_workers` default 0 / `autoscaling_max_workers` default 10, `infra/gcp/variables.tf:137-147`) | Spark dynamic allocation on (min 2 / max 20 *executors*, `conf/base/parameters/platform.yml:29-32`), AQE on — distinct from Dataproc autoscaling (*workers*) | Cloud scoring at 100M rows |
| AWS EMR | Stub off by default; for 100M rows set `enable_emr = true` and provision autoscaling (commented `aws_emr_auto_scaling_policy` in `infra/aws/main.tf`, CORE capacity `emr_core_min/max_capacity`) | Same Spark conf via `spark_params` | Cloud scoring at 100M rows |

BigQuery/Athena tables backing the catalog are already
partitioned (`partition_field = run_date`, DAY) and clustered on
`score_model_id` (see `conf/base/catalog.yml`) — keep new tables
partitioned/clustered the same way so 100M-row scans stay cheap.
Watch per-worker billing while clusters run; destroy/idle after a
run.

## Phase 3: retraining scheduler + alerts

| Cloud | Resource | Gate |
|---|---|---|
| GCP | `google_cloud_scheduler_job.retraining` (cron `retraining_schedule_cron`, default weekly) | `count = var.enable_retraining_scheduler ? 1 : 0`, default **false** — no scheduled retraining until enabled; set `retraining_job_uri` to a real endpoint first |
| GCP | `google_pubsub_topic.alerts` (alerts sink; email subscription is a commented stub) | Always created (topic only, no cost when idle); uncomment `google_pubsub_subscription.alerts_email` + set `alerts_email` for email delivery |
| AWS | `aws_cloudwatch_event_rule.retraining` (EventBridge, weekly) | **Commented stub** — uncomment + set a real target to provision; costs nothing while commented |

Scheduler cadence mirrors `retraining_params.schedule_days = 7` in
`conf/base/parameters/platform.yml`. Monitoring alerts land in the `alerts`
catalog dataset and, on GCP, the Pub/Sub topic above.

## Encryption (CMEK) — production MUST set

Buckets use Google-/AWS-managed encryption by default (dev). That is
an accepted dev default only. **In production you MUST set a
customer-managed key:** GCP `kms_key_name` (full KMS key path,
`infra/gcp/variables.tf:121-130`) or AWS `kms_key_id`
(`infra/aws/variables.tf:96-105`). Empty means managed encryption and
must not ship to production. Validation enforces it (D7/X7): set
`production = true` and `terraform plan` fails if the CMEK key is
empty (`!var.production || var.kms_key_name != ""` /
`!var.production || var.kms_key_id != ""`).

**Artifacts bucket override (MLops R2, `artifacts_bucket` param):**
set the Terraform `artifacts_bucket` variable and the matching
`params:artifacts_bucket` runtime param together (default
`'forecast-ml-artifacts'`, `conf/base/parameters/platform.yml:51-61`);
the overlays interpolate `${params:artifacts_bucket}` in every
`filepath`, so that one param repoints the pipeline.

**Bucket lifecycle (GCP):** the GCP artifacts bucket expires
noncurrent versions only after 90 days (`lifecycle_rule` at
`infra/gcp/main.tf:94-102`, `age = 90` with
`with_state = "ARCHIVED"` — GCS's name for noncurrent versions;
live/current objects are never expired by this rule), while the
temp bucket purges current + noncurrent objects after 7 days
(`with_state = "ANY"`, `infra/gcp/main.tf:134-142`).

## Validation

```bash
terraform fmt -check -recursive infra/
terraform validate   # run inside infra/gcp and infra/aws after init
```
