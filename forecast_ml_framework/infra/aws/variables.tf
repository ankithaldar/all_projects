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
  description = "S3 bucket for Athena query results. Defaults to a dedicated bucket (forecast-ml-athena-results, created by this stack) so query results never land in the artifacts bucket (MLops-15). Set an existing bucket name only to reuse a pre-provisioned results bucket."
  type        = string
  default     = "forecast-ml-athena-results"
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

variable "emr_service_role_arn" {
  description = "ARN of the scoped IAM role EMR assumes as its service role. Required when enable_emr is true (enforced by the validation block below). Supply a least-privilege role for this cluster -- never a shared/broad role. The stack intentionally does not look up or create any default role."
  type        = string
  default     = null

  validation {
    condition     = !var.enable_emr || (var.emr_service_role_arn != null && var.emr_service_role_arn != "")
    error_message = "emr_service_role_arn must be set when enable_emr is true."
  }
}

variable "emr_ec2_instance_profile_arn" {
  description = "ARN of the EC2 instance profile for EMR nodes (required by the aws provider ~> 5.0 schema when enable_emr is true; enforced by the validation block below). Supply a least-privilege profile for this cluster."
  type        = string
  default     = null

  validation {
    condition     = !var.enable_emr || (var.emr_ec2_instance_profile_arn != null && var.emr_ec2_instance_profile_arn != "")
    error_message = "emr_ec2_instance_profile_arn must be set when enable_emr is true."
  }
}

# --- CMEK for artifacts buckets (Antisec F-12 / R-06) -------------------------
# KMS key ID or ARN for SSE-KMS on the artifacts/Athena-results buckets.
# Empty string (default) uses AES256 (dev); production must set a real
# customer-managed KMS key (deny-public + CMEK is the F-12 control).
# The production flag below enforces that precondition at plan time
# (D7): production = true without a key fails validation, so CMEK
# can never be silently optional in production.

variable "production" {
  description = "Mark this deployment as production. When true, kms_key_id must be set (enforced by the validation block below). Default false (dev)."
  type        = bool
  default     = false
}

variable "kms_key_id" {
  description = "KMS key ID/ARN for SSE-KMS on the artifacts and Athena-results buckets. Empty string uses AES256 (dev default); production must set a customer-managed KMS key."
  type        = string
  default     = ""

  validation {
    condition     = !var.production || var.kms_key_id != ""
    error_message = "kms_key_id must be set when production is true (CMEK is required in production; set a customer-managed KMS key ID/ARN)."
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
