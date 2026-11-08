# Azure variables. No credentials, subscription IDs, or tenant IDs
# are hardcoded anywhere in this module: subscription_id and
# tenant_id are required variables with no defaults (pass via
# -var or TF_VAR_*, see README.md). Storage / vault name defaults
# match conf/azure/catalog.yml, which resolves every filepath via
# the single $${params:artifacts_blob_url} interpolation (Terraform
# output artifacts_blob_url).

variable "subscription_id" {
  description = "Azure subscription ID. Required - no default (pass via -var or TF_VAR_subscription_id)."
  type        = string
}

variable "tenant_id" {
  description = "Azure AD (Entra ID) tenant ID for the v2.0 OIDC issuer (https://login.microsoftonline.com/<tenant>/v2.0). Required - no default."
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group for all Azure resources."
  type        = string
  default     = "forecast-ml-rg"
}

variable "location" {
  description = "Azure region for regional resources. Non-binding: this variable makes no residency or data-boundary claim (D-SECRETS-PII-GDPR-1 section 5.2(e)); the residency decision is taken at P2 IaC before P11 and artifacts created before then carry no residency guarantee."
  type        = string
  default     = "eastus2"
}

variable "artifacts_storage_account" {
  description = "Name of the storage account for pipeline artifacts. Matches conf/azure/catalog.yml via the artifacts_blob_url output. Lowercase alnum, 3-24 chars, no hyphens (Azure storage account rule)."
  type        = string
  default     = "forecastmlartifacts"

  validation {
    condition     = can(regex("^[a-z0-9]{3,24}$", var.artifacts_storage_account))
    error_message = "artifacts_storage_account must match ^[a-z0-9]{3,24}$ (lowercase alnum, 3-24 chars, no hyphens)."
  }
}

variable "key_vault_name" {
  description = "Name of the Key Vault. Alphanumeric, 3-24 chars, no hyphens in the frozen default (K8s/CLI safety); matches the ^[a-z0-9]{3,24}$ rule enforced below."
  type        = string
  default     = "forecastmlvault"

  validation {
    condition     = can(regex("^[a-z0-9]{3,24}$", var.key_vault_name))
    error_message = "key_vault_name must match ^[a-z0-9]{3,24}$ (lowercase alnum, 3-24 chars, no hyphens)."
  }
}

variable "vnet_cidr" {
  description = "Virtual network address space (CIDR). Non-binding, same no-residency-pin rule as var.location."
  type        = string
  default     = "10.0.0.0/16"
}

variable "aks_subnet_cidr" {
  description = "AKS subnet address prefix (CIDR) inside var.vnet_cidr."
  type        = string
  default     = "10.0.1.0/24"
}

variable "private_subnet_cidr" {
  description = "Private-endpoints subnet address prefix (CIDR) inside var.vnet_cidr."
  type        = string
  default     = "10.0.2.0/24"
}

variable "api_audience" {
  description = "Separate API audience (Entra app registration base URI) for the serving API. Must differ from the workload-identity audience api://AzureADTokenExchange (enforced below)."
  type        = string
  default     = "api://forecast-ml-api"

  validation {
    condition     = var.api_audience != "api://AzureADTokenExchange"
    error_message = "api_audience must differ from the workload-identity audience api://AzureADTokenExchange."
  }
}

variable "enable_aks" {
  description = "Create the AKS cluster stub. Default false (count = 0) - off by default per the Dataproc/EMR precedent; set true only when needed."
  type        = bool
  default     = false
}

variable "kms_key_id" {
  description = "Key Vault key ID for storage CMEK (customer-managed encryption key). Empty string (default) uses Microsoft-managed encryption (dev); production must set a real CMEK key (see README.md)."
  type        = string
  default     = ""

  validation {
    condition     = !var.production || var.kms_key_id != ""
    error_message = "kms_key_id must be set when production is true (CMEK is required in production; set a customer-managed encryption key ID)."
  }
}

variable "production" {
  description = "Mark this deployment as production. When true, kms_key_id must be set (enforced by the validation block above). Default false (dev)."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags applied to supported resources. Never secrets."
  type        = map(string)
  default = {
    Project   = "forecast-ml-framework"
    ManagedBy = "terraform"
    Purpose   = "dev"
  }
}
