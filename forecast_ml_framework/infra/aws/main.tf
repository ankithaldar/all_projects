# AWS infrastructure for forecast-ml-framework (Phase 2, config-ready).
#
# Creates: S3 artifacts bucket (versioning on), Athena database, and an
# EMR cluster stub gated by var.enable_emr (count = 0 by default).
# Auth comes from the environment only (AWS_PROFILE / AWS_* env vars);
# nothing is hardcoded here.

terraform {
  required_version = ">= 1.3"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  # Dedicated Athena results bucket (MLops-15): query results land
  # in their own bucket with its own lifecycle/encryption boundary,
  # never mixed into the artifacts bucket.
  athena_results_bucket = aws_s3_bucket.athena_results.id
}

# --- Artifacts bucket ---------------------------------------------------------
# Holds everything conf/aws/catalog.yml points at
# (s3://forecast-ml-artifacts/data/..., .../artifacts/...).

resource "aws_s3_bucket" "artifacts" {
  bucket = var.artifacts_bucket
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      # CMEK (F-12): customer-managed KMS key when var.kms_key_id
      # is set; AES256 otherwise (dev default -- production must
      # set kms_key_id, see variables.tf).
      sse_algorithm     = var.kms_key_id != "" ? "aws:kms" : "AES256"
      kms_master_key_id = var.kms_key_id != "" ? var.kms_key_id : null
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Deny-public hardening (F-12): explicit bucket policy denying
# non-TLS access. Combined with the public-access block above, no
# anonymous/public principal can read or write this bucket.
resource "aws_s3_bucket_policy" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.artifacts.arn,
          "${aws_s3_bucket.artifacts.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}

# Bucket lifecycle (MLops-14): without these rules old object
# versions and abandoned multipart uploads are billed forever.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# --- Athena results bucket (dedicated, MLops-15) ------------------------------
# Query results get their own encryption/lifecycle boundary instead
# of sharing the artifacts bucket.

resource "aws_s3_bucket" "athena_results" {
  bucket = var.athena_results_bucket
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "athena_results" {
  bucket = aws_s3_bucket.athena_results.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "athena_results" {
  bucket = aws_s3_bucket.athena_results.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.kms_key_id != "" ? "aws:kms" : "AES256"
      kms_master_key_id = var.kms_key_id != "" ? var.kms_key_id : null
    }
  }
}

resource "aws_s3_bucket_public_access_block" "athena_results" {
  bucket                  = aws_s3_bucket.athena_results.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "athena_results" {
  bucket = aws_s3_bucket.athena_results.id

  rule {
    id     = "expire-results"
    status = "Enabled"

    filter {}

    expiration {
      days = 90
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# --- Athena database ----------------------------------------------------------
# Queryable staging/metrics tables over the artifacts bucket.

resource "aws_athena_database" "forecast" {
  name   = var.athena_database
  bucket = local.athena_results_bucket
}

# --- EMR cluster stub (disabled by default) -----------------------------------
# count = var.enable_emr ? 1 : 0, and var.enable_emr defaults to false,
# so nothing is created (and nothing is billed) until explicitly enabled.
# To enable: set enable_emr = true, emr_subnet_id to a subnet in
# var.aws_region, and emr_service_role_arn to the ARN of a scoped
# IAM role for this cluster. This is a stub for config parity with
# the GCP Dataproc cluster — it is not a tuned production EMR setup.
# NOTE (MLops-8, Antisec F-07): service_role must be a role ARN,
# never a bare role name -- the AWS provider rejects bare names at
# apply. This stack neither creates nor looks up any default/shared
# role: the operator supplies a least-privilege role for this
# cluster via var.emr_service_role_arn (required when enabled,
# enforced by its validation block). See infra/aws/README.md.

resource "aws_emr_cluster" "stub" {
  count         = var.enable_emr ? 1 : 0
  name          = "forecast-ml-emr-stub"
  release_label = var.emr_release_label
  applications  = ["Spark"]
  service_role  = var.emr_service_role_arn

  ec2_attributes {
    subnet_id        = var.emr_subnet_id
    instance_profile = var.emr_ec2_instance_profile_arn
  }

  # aws provider ~> 5.0 schema: legacy master_instance_type /
  # instance_count arguments were removed -- instance groups are
  # required. Master-only stub (parity with the old
  # instance_count = 1): one master, zero core workers. To run
  # real Spark work, raise the core count (or uncomment the
  # autoscaling stub below) -- that is billed per instance-hour.
  master_instance_group {
    instance_type  = var.emr_master_instance_type
    instance_count = 1
  }

  core_instance_group {
    instance_type  = var.emr_master_instance_type
    instance_count = 0
  }

  log_uri = "s3://${aws_s3_bucket.artifacts.id}/emr-logs/"

  tags = var.tags
}

# --- Retraining EventBridge rule stub (Phase 3, COMMENTED OUT) ---------------
# Provision note: this stack ships the retraining schedule as a
# commented stub, not an applied resource. To provision it, uncomment
# the rule below and set a real target (e.g. a Lambda / Step
# Functions / CodeBuild retraining trigger). It mirrors the GCP
# google_cloud_scheduler_job (retraining_params.schedule_days = 7,
# weekly cron) and costs nothing while commented. COST NOTE: when
# provisioned, each firing triggers a retraining run billed per run.
#
# resource "aws_cloudwatch_event_rule" "retraining" {
#   name                = "forecast-ml-retraining"
#   description         = "Phase 3 retraining scheduler stub (weekly, mirrors GCP Cloud Scheduler)."
#   schedule_expression = "cron(0 6 ? * MON *)"  # weekly, matches retraining_params.schedule_days = 7
#
#   tags = var.tags
# }
#
# # Target stub (uncomment with a real retraining target):
# # resource "aws_cloudwatch_event_target" "retraining" {
# #   rule = aws_cloudwatch_event_rule.retraining.name
# #   arn  = "arn:aws:lambda:us-east-1:ACCOUNT_ID:function:forecast-ml-retraining"  # set a real ARN
# # }

# --- EMR autoscaling stub (100M-row cloud scaling, COMMENTED OUT) -----------
# Provision note: this stack ships autoscaling as a commented stub,
# not an applied resource. To provision it, uncomment the resource
# below (and add CORE instance groups to aws_emr_cluster.stub) --
# see infra/aws/README.md, "100M-row sizing / EMR autoscaling".
# It scales CORE capacity between var.emr_core_min/max_capacity on
# YARN memory pressure. COST NOTE: scaled instances are billed per
# instance-hour while the cluster runs; idle/destroy after a run.
#
# resource "aws_emr_auto_scaling_policy" "core" {
#   name                = "forecast-ml-core-scaling"
#   cluster_id          = aws_emr_cluster.stub[0].id
#   instance_group_type = "CORE"
#
#   simple_config {
#     name = "ScaleOutOnMemoryPressure"
#
#     schedule {
#       adjust_type          = "CHANGE_IN_CAPACITY"
#       scaling_adjustment   = 1
#       adjustment_origin {
#         timestamp = "2026-11-05T00:00:00Z"  # set to a real time when uncommenting
#       }
#     }
#
#     trigger {
#       cloudwatch_alarm_definition {
#         comparison_operator = "GREATER_THAN"
#         evaluation_periods  = 1
#         metric_name         = "YARNMemoryAvailablePercentage"
#         namespace           = "AWS/ElasticMapReduce"
#         period              = 300
#         statistic           = "AVERAGE"
#         threshold           = 15.0
#         unit                = "PERCENT"
#         dimensions = {
#           JobFlowId = aws_emr_cluster.stub[0].id
#         }
#       }
#     }
#   }
# }
