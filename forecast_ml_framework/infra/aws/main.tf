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
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  athena_results_bucket = var.athena_results_bucket == "" ? var.artifacts_bucket : var.athena_results_bucket
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
      sse_algorithm = "AES256"
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

# --- Athena database ----------------------------------------------------------
# Queryable staging/metrics tables over the artifacts bucket.

resource "aws_athena_database" "forecast" {
  name   = var.athena_database
  bucket = local.athena_results_bucket
}

# --- EMR cluster stub (disabled by default) -----------------------------------
# count = var.enable_emr ? 1 : 0, and var.enable_emr defaults to false,
# so nothing is created (and nothing is billed) until explicitly enabled.
# To enable: set enable_emr = true and emr_subnet_id to a subnet in
# var.aws_region. This is a stub for config parity with the GCP Dataproc
# cluster — it is not a tuned production EMR setup.
# NOTE: service_role "EMR_DefaultRole" must PRE-EXIST in the AWS account
# (the default EMR service-linked role, created automatically on first
# EMR use or by AWS). This stack does NOT create the IAM role; if it does
# not exist, cluster creation fails. See infra/aws/README.md. A subnet
# is required when enable_emr is true (enforced by the validation block
# on var.emr_subnet_id).

resource "aws_emr_cluster" "stub" {
  count         = var.enable_emr ? 1 : 0
  name          = "forecast-ml-emr-stub"
  release_label = var.emr_release_label
  applications  = ["Spark"]
  service_role  = "EMR_DefaultRole"

  ec2_attributes {
    subnet_id = var.emr_subnet_id
  }

  master_instance_type = var.emr_master_instance_type
  instance_count       = 1

  log_uri = "s3://${aws_s3_bucket.artifacts.id}/emr-logs/"

  tags = var.tags
}

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
