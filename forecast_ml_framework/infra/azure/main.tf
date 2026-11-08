# Azure infrastructure for forecast-ml-framework (P2-I1).
#
# Creates: resource group, private storage account (versioning on)
# + data/models/metrics containers, private Key Vault (references
# only - secret bindings are P2-S1 secrets.tf, no secret value
# here), workload managed identity + Entra federated credential
# for AKS workload identity, and an AKS stub cluster (off by
# default). Network / private endpoints live in network.tf; the
# provider block lives in providers.tf.
# Auth comes from the environment only; nothing is hardcoded here.

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location

  tags = var.tags
}

# --- Artifacts storage -----------------------------------------------------
# Private by default: public network access disabled, shared key
# disabled, TLS 1.2 minimum, network rules deny by default.

resource "azurerm_storage_account" "artifacts" {
  name                     = var.artifacts_storage_account
  resource_group_name      = azurerm_resource_group.main.name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  public_network_access_enabled     = false
  allow_nested_items_to_be_public   = false
  min_tls_version                   = "TLS1_2"
  shared_access_key_enabled         = false
  cross_tenant_replication_enabled  = false
  infrastructure_encryption_enabled = true
  default_to_oauth_authentication   = true

  network_rules {
    default_action = "Deny"
    bypass         = ["None"]
  }

  blob_properties {
    versioning_enabled = true
  }

  # CMEK: customer-managed key when var.kms_key_id is set;
  # Microsoft-managed encryption otherwise (dev default -
  # production must set kms_key_id, see variables.tf).
  dynamic "customer_managed_key" {
    for_each = var.kms_key_id != "" ? [1] : []
    content {
      key_vault_key_id          = var.kms_key_id
      user_assigned_identity_id = azurerm_user_assigned_identity.workloads.id
    }
  }

  tags = var.tags
}

resource "azurerm_storage_container" "data" {
  name                  = "data"
  storage_account_name  = azurerm_storage_account.artifacts.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.artifacts.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "metrics" {
  name                  = "metrics"
  storage_account_name  = azurerm_storage_account.artifacts.name
  container_access_type = "private"
}

# --- Key Vault (references only) -------------------------------------------
# Private by default. Secret bindings (secret references, never
# values) are owned by P2-S1 infra/azure/secrets.tf.

resource "azurerm_key_vault" "main" {
  name                          = var.key_vault_name
  location                      = var.location
  resource_group_name           = azurerm_resource_group.main.name
  tenant_id                     = var.tenant_id
  sku_name                      = "standard"
  purge_protection_enabled      = true
  soft_delete_retention_days    = 90
  public_network_access_enabled = false
  enable_rbac_authorization     = true

  network_acls {
    default_action = "Deny"
    bypass         = "None"
  }

  tags = var.tags
}

# --- Workload identity + Entra federated credential ------------------------
# AKS workload identity federates Entra tokens: audience is the
# workload-identity audience api://AzureADTokenExchange (never the
# API audience var.api_audience), subject is the forecast-ml
# service account, issuer is the Entra v2.0 endpoint.

# azurerm ~> 3.0 names this azurerm_user_assigned_identity (the
# azurerm_managed_identity type is the post-3.x rename); the
# "workloads" label and all federated-credential fields below are
# exactly as frozen in P2-I1.
resource "azurerm_user_assigned_identity" "workloads" {
  name                = "forecast-ml-workloads"
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name

  tags = var.tags
}

resource "azurerm_federated_identity_credential" "aks" {
  name                = "forecast-ml-aks-federated"
  resource_group_name = azurerm_resource_group.main.name
  parent_id           = azurerm_user_assigned_identity.workloads.id

  audience = ["api://AzureADTokenExchange"]
  issuer   = "https://login.microsoftonline.com/${var.tenant_id}/v2.0"
  subject  = "system:serviceaccount:forecast-ml:forecast-ml"
}

# --- AKS cluster stub ------------------------------------------------------
# Created on enable only (count = var.enable_aks ? 1 : 0, default
# false per the Dataproc/EMR off-by-default precedent). OIDC issuer
# + workload identity are enabled so the federated credential above
# can be consumed by AKS workloads.

resource "azurerm_kubernetes_cluster" "aks" {
  count               = var.enable_aks ? 1 : 0
  name                = "forecast-ml-aks"
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "forecast-ml-aks"

  oidc_issuer_enabled       = true
  workload_identity_enabled = true

  default_node_pool {
    name       = "default"
    node_count = 1
    vm_size    = "Standard_B2s"
  }

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}
