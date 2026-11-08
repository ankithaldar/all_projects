# Azure providers for forecast-ml-framework (P2-I1).
#
# Terraform >= 1.3, hashicorp/azurerm ~> 3.0. Auth comes from the
# environment only (ARM_SUBSCRIPTION_ID / ARM_TENANT_ID /
# ARM_CLIENT_ID / ARM_CLIENT_SECRET or the Azure CLI); nothing is
# hardcoded here. subscription_id / tenant_id are required
# variables with no defaults (see variables.tf).

terraform {
  required_version = ">= 1.3"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  subscription_id = var.subscription_id
  tenant_id       = var.tenant_id

  features {}
}
