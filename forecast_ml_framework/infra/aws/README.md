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

**EMR prerequisites (validation + docs, chosen over creating an IAM
role resource):** `emr_subnet_id` is **required** when
`enable_emr = true` — enforced by a Terraform `validation` block on
`var.emr_subnet_id` (plan fails if unset). The stub uses
`service_role = "EMR_DefaultRole"`, which **must pre-exist** in the
AWS account (the default EMR service-linked role, auto-created on
first EMR use). This stack does not create that IAM role; if it does
not exist, EMR cluster creation will fail with an IAM error — create
/ enable the default role out of band before enabling.

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
  = 1000`, dynamic allocation on, AQE on in
  `conf/base/parameters.yml`) — ~100k rows/partition at 100M rows.
- **Athena:** keep staging tables partitioned (by `run_date`) the
  way the catalog does, so 100M-row scans prune partitions.
- **Cost note:** EMR is **off by default** (`enable_emr = false`);
  when enabled (and autoscaling provisioned), instances are billed
  per instance-hour while the cluster runs — idle/destroy after a
  run.

## State

Local state by default (`terraform.tfstate`, gitignored — do not commit it).
For shared use, configure a remote backend with locking (e.g. S3 backend with
a DynamoDB lock table) in a `backend.tf` added separately.

## Outputs → conf overlays

- `artifacts_bucket_url` (`s3://forecast-ml-artifacts`) — already the default
  in `conf/aws/catalog.yml`; if you override `artifacts_bucket`, update the
  `filepath` values there to match.
- `athena_database` — Athena database name for staging/metrics tables.
- `emr_cluster_id` — empty unless `enable_emr = true`.

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
