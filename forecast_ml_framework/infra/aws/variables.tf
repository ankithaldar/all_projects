# AWS variables. No credentials, account IDs, or keys are hardcoded
# anywhere in this module: auth comes from the environment only
# (AWS_PROFILE / AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY — see
# README.md). Bucket defaults match conf/aws/catalog.yml
# (s3://forecast-ml-artifacts/...).

variable "aws_region" {
  description = "AWS region for regional resources."
  type        = string
  default     = "us-east-1"
}

variable "artifacts_bucket" {
  description = "Name of the S3 bucket for pipeline artifacts. Matches conf/aws/catalog.yml (s3://forecast-ml-artifacts/...)."
  type        = string
  default     = "forecast-ml-artifacts"
}

variable "athena_database" {
  description = "Athena database name for queryable staging/metrics tables."
  type        = string
  default     = "forecast_ml"
}

variable "athena_results_bucket" {
  description = "S3 bucket for Athena query results. Empty string reuses the artifacts bucket."
  type        = string
  default     = ""
}

variable "enable_emr" {
  description = "Create the EMR cluster stub. Default false (count = 0) — enable only when needed."
  type        = bool
  default     = false
}

variable "emr_release_label" {
  description = "EMR release label for the stub cluster."
  type        = string
  default     = "em-7.5.0"
}

variable "emr_master_instance_type" {
  description = "EC2 instance type for the EMR master node (stub)."
  type        = string
  default     = "m5.xlarge"
}

variable "emr_subnet_id" {
  description = "Subnet ID for the EMR stub cluster. Required when enable_emr is true; unused otherwise."
  type        = string
  default     = null

  validation {
    condition     = !var.enable_emr || (var.emr_subnet_id != null && var.emr_subnet_id != "")
    error_message = "emr_subnet_id must be set when enable_emr is true."
  }
}

# --- EMR autoscaling stub (100M-row cloud scaling, commented in main.tf) -----
# These only take effect if the commented aws_emr_auto_scaling_policy
# resource in main.tf is uncommented (and enable_emr = true). COST
# NOTE: autoscaled CORE instances are billed per instance-hour while
# the cluster runs -- idle/destroy the cluster after a run.

variable "emr_core_min_capacity" {
  description = "Minimum CORE instance-group capacity for the (commented) EMR autoscaling policy."
  type        = number
  default     = 1
}

variable "emr_core_max_capacity" {
  description = "Maximum CORE instance-group capacity for the (commented) EMR autoscaling policy. Bounds cost; raise only for 100M-row runs."
  type        = number
  default     = 10
}

variable "tags" {
  description = "Tags applied to supported resources."
  type        = map(string)
  default = {
    Project   = "forecast-ml-framework"
    ManagedBy = "terraform"
    Purpose   = "dev"
  }
}
