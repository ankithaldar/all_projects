# Azure secret reference for forecast-ml-framework (P2-S1).
#
# Reference only: the data source below reads the named
# Key Vault secret shell by name. No value is stored here
# and no value-bearing resource is declared, so nothing
# secret reaches Terraform state. Fill the value out-of-band
# with the Azure CLI after apply, for example:
#   az keyvault secret set --vault-name <vault> --name <name> --value <value>
# The secret name is a variable (default below) so each
# environment can point at its own shell. Auth comes from
# the environment only (ARM_* env vars or the Azure CLI);
# nothing is hardcoded here.

variable "secret_name" {
  description = "Name of the Key Vault secret shell. The value is set out-of-band via the Azure CLI, never in Terraform."
  type        = string
  default     = "forecast-ml-api-secrets"
}

data "azurerm_key_vault_secret" "main" {
  name         = var.secret_name
  key_vault_id = azurerm_key_vault.main.id
}

output "secret_id" {
  description = "ID of the Key Vault secret shell (name reference only, never a value)."
  value       = data.azurerm_key_vault_secret.main.id
}
