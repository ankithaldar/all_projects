# AWS infrastructure — `infra/aws/` (config-ready)

Terraform for the AWS side of forecast-ml-framework (Phase 2). This stack is
**config-ready**: it applies cleanly in principle, but the EMR stub is off by
default, so the only always-on cost is the S3 bucket.

## What it creates

| Resource | Purpose | Conf mapping |
|---|---|---|
| `aws_s3_bucket.artifacts` + versioning + SSE + public-access-block | Pipeline artifacts bucket | `s3://forecast-ml-artifacts/...` in `conf/aws/catalog.yml` |
| `aws_athena_database.forecast` | Queryable staging/metrics tables | Athena over the artifacts bucket |
| `aws_emr_cluster.stub` (`count = var.enable_emr ? 1 : 0`, default **false**) | EMR stub (config parity with GCP Dataproc) | Spark jobs, when enabled |
| `aws_cloudwatch_event_rule.retraining` (**commented stub**, EventBridge) | Retraining scheduler stub (weekly cron) | Mirrors GCP `google_cloud_scheduler_job`; uncomment + set a real target to provision |

Defaults match `conf/` (`artifacts_bucket = "forecast-ml-artifacts"`,
`athena_database = "forecast_ml"`). No account IDs, credentials, or keys are
hardcoded — everything comes from variables and the environment.

## Prerequisites

- Terraform `>= 1.3` (`terraform version`).
- AWS credentials from the environment only — set `AWS_PROFILE` (named
  profile) or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+
  `AWS_SESSION_TOKEN` if temporary). Never put credentials in `.tf`,
  `.tfvars`, or `conf/`.

## Apply order

Run from this directory:

```bash
terraform init
terraform plan
terraform apply
```

Full lifecycle:

```bash
terraform plan
terraform apply
terraform destroy
```

To enable the EMR stub (off by default):

```bash
terraform apply -var="enable_emr=true" -var="emr_subnet_id=subnet-xxxxxxxx"
```

**EMR prerequisites (validation + docs):** `emr_subnet_id` and
`emr_service_role_arn` are **required** when `enable_emr = true` —
enforced by Terraform `validation` blocks (plan fails if unset).
The stub uses `service_role = var.emr_service_role_arn` (ARN of a
scoped IAM role) plus `var.emr_ec2_instance_profile_arn`
(`infra/aws/main.tf:203`, `infra/aws/variables.tf:60-67`;
W5/MLops-8). The old bare `EMR_DefaultRole` string no longer
ships (previously prose said it must pre-exist; that requirement
is gone — now the ARN variable must pre-exist). This stack
does not create that IAM role; provision it out of band before
enabling.

## 100M-row sizing / EMR autoscaling

Local pandas is capped by `pandas_conversion_params.max_rows_for_pandas`
(default 5000000); 100M-row scoring must use the Spark path
(`scoring_params.spark_scoring.enabled = true`,
`partition_batch_size = 10000`, `max_rows = null` = unbounded).

- **Worker tiers:** dev = EMR stub off (`enable_emr = false`, no
  cost); 100M rows = `enable_emr = true` with CORE instance groups
  sized to the job.
- **Autoscaling provision:** autoscaling ships as a **commented
  stub** — `aws_emr_auto_scaling_policy.core` in `main.tf` (search
  for `aws_emr_auto_scaling_policy`). To provision: add CORE
  instance groups to `aws_emr_cluster.stub`, uncomment the policy,
  and set `emr_core_min_capacity` / `emr_core_max_capacity`
  (defaults 1 / 10 — the max bounds cost). It scales CORE capacity
  out on YARN memory pressure.
- **Shuffle:** same `spark_params` as GCP (`spark.sql.shuffle.partitions
  = 1000`, dynamic allocation on (min 2 / max 20 executors), AQE on in
  `conf/base/parameters/platform.yml`) — ~100k rows/partition at 100M rows.
- **Athena:** keep staging tables partitioned (by `run_date`) the
  way the catalog does, so 100M-row scans prune partitions.
- **Cost note:** EMR is **off by default** (`enable_emr = false`);
  when enabled (and autoscaling provisioned), instances are billed
  per instance-hour while the cluster runs — idle/destroy after a
  run.

## Phase 3: retraining EventBridge stub

The retraining schedule ships as a **commented stub** —
`aws_cloudwatch_event_rule.retraining` in `main.tf` (search for
`aws_cloudwatch_event_rule`, EventBridge). It mirrors the GCP
`google_cloud_scheduler_job` (weekly, matching
`retraining_params.schedule_days = 7` in
`conf/base/parameters/platform.yml`). To provision: uncomment the rule and the
`aws_cloudwatch_event_target` stub, and set the target to a real
retraining trigger (Lambda / Step Functions / CodeBuild ARN). While
commented it creates nothing and costs nothing; when provisioned,
each firing triggers a retraining run billed per run.
Monitoring alerts on AWS land in the `alerts` catalog dataset
(`artifacts/monitoring/alerts.parquet`); route them to SNS/email in
a separate change if needed.

## State

Local state by default (`terraform.tfstate`, gitignored — do not commit it).
For shared use, adopt the shipped `aws/backend.tf.example` (S3 +
DynamoDB, A6) by copying it to `backend.tf` and running
`terraform init -migrate-state`.

## Outputs → conf overlays

- `artifacts_bucket_url` (`s3://forecast-ml-artifacts`) — already the default
  in `conf/aws/catalog.yml`; if you override `artifacts_bucket`, set
  the matching `params:artifacts_bucket` runtime param
  (`conf/base/parameters/platform.yml:51-61`, see `conf/README.md`).
- `athena_database` — Athena database name for staging/metrics tables.
- `emr_cluster_id` — empty unless `enable_emr = true`.

## Encryption (CMEK) — production MUST set

- **CMEK — production MUST set `kms_key_id`** (R-06): empty (default)
  uses AWS-managed `AES256`; setting it switches the artifacts and
  Athena-results buckets to `aws:kms` (`infra/aws/main.tf:52-56,139-140`,
  `infra/aws/variables.tf:96-105`). Production MUST set a real KMS key
  ARN. Validation enforces it (D7/X7): `production = true` with an
  empty `kms_key_id` fails `terraform plan`.

## Costs

- S3: storage + requests for the artifacts bucket (versioning keeps old
  object versions — monitor storage growth).
- Athena: per-query scan cost against the results bucket.
- EMR: **off by default (`enable_emr = false`, count = 0), so no EMR cost
  until explicitly enabled.** When enabled, EMR is billed per cluster-hour
  plus EC2 — destroy it when not in use.

## Validation

```bash
terraform fmt -check -recursive .
terraform validate
```
