# Azure outputs. Feed these into conf overlays / runtime params:
# artifacts_blob_url        -> $${params:artifacts_blob_url}
#                              (conf/azure/catalog.yml, ONE
#                              interpolation mechanism)
# key_vault_name            -> Key Vault name (references only)
# aks_cluster_name           -> AKS stub name (empty when off)
# private_endpoint_ips      -> private endpoint IPs (storage+vault)
# workload_identity_client_id -> managed identity client ID
# oidc_issuer_url           -> Entra v2.0 issuer URL
#
# Conditional outputs use try(...[0], "") so terraform plan with
# enable_aks = false still emits every output.

output "artifacts_blob_url" {
  description = "Blob URL of the artifacts storage account (matches conf/azure/catalog.yml via $${params:artifacts_blob_url})."
  value       = "https://${azurerm_storage_account.artifacts.name}.blob.core.windows.net"
}

output "key_vault_name" {
  description = "Name of the Key Vault (references only, never secret values)."
  value       = azurerm_key_vault.main.name
}

output "aks_cluster_name" {
  description = "Name of the AKS stub cluster (empty when enable_aks is false)."
  value       = try(azurerm_kubernetes_cluster.aks[0].name, "")
}

output "private_endpoint_ips" {
  description = "Private IPs of the storage and vault private endpoints."
  value = [
    try(azurerm_private_endpoint.storage.private_service_connection[0].private_ip_address, ""),
    try(azurerm_private_endpoint.vault.private_service_connection[0].private_ip_address, ""),
  ]
}

output "workload_identity_client_id" {
  description = "Client ID of the workload managed identity (empty if unavailable)."
  value       = try(azurerm_user_assigned_identity.workloads.client_id, "")
}

output "oidc_issuer_url" {
  description = "Entra ID v2.0 OIDC issuer URL for workload identity."
  value       = "https://login.microsoftonline.com/${var.tenant_id}/v2.0"
}
