# Azure infrastructure — `infra/azure/`

Terraform for the Azure side of forecast-ml-framework (P2-I1).
Mirrors the GCP/AWS stacks (`infra/gcp/`, `infra/aws/`) with Azure
parity: private storage + Key Vault, VNet with private endpoints,
and Entra OIDC federated credentials for AKS workload identity.

## What it creates

| Resource | Purpose | Conf mapping |
|---|---|---|
| `azurerm_storage_account.artifacts` (versioning on, private) | Pipeline artifacts storage | `${params:artifacts_blob_url}/...` in `conf/azure/catalog.yml` |
| `azurerm_storage_container.data` / `models` / `metrics` (private) | Artifact prefixes | Same account, per-prefix containers |
| `azurerm_key_vault.main` (private, references only) | Secret references (values are P2-S1 `secrets.tf`, never here) | Key Vault name output |
| `azurerm_virtual_network.main` | VNet for AKS + private endpoints | — |
| `azurerm_subnet.aks` + `azurerm_subnet.private` | AKS subnet / private-endpoints subnet | — |
| `azurerm_network_security_group.main` | Deny-public NSG, allows only `8000` from the private subnet | — |
| `azurerm_private_dns_zone.blob` (`privatelink.blob.core.windows.net`) + VNet link | Private DNS for storage | — |
| `azurerm_private_dns_zone.vault` (`privatelink.vaultcore.azure.net`) + VNet link | Private DNS for vault | — |
| `azurerm_private_endpoint.storage` + `azurerm_private_endpoint.vault` | Private links for storage + vault | `private_endpoint_ips` output |
| `azurerm_user_assigned_identity.workloads` (azurerm ~> 3.0 name for the workload managed identity) | Workload identity | `workload_identity_client_id` output |
| `azurerm_federated_identity_credential.aks` | Entra OIDC for AKS workload identity (`audience = ['api://AzureADTokenExchange']`, `subject = 'system:serviceaccount:forecast-ml:forecast-ml'`, issuer `https://login.microsoftonline.com/<tenant>/v2.0`) | `oidc_issuer_url` output |
| `azurerm_kubernetes_cluster.aks` (`count = var.enable_aks ? 1 : 0`, default **false**) | AKS stub cluster | `aks_cluster_name` output |

Storage and vault are private by default:
`public_network_access_enabled = false`,
`shared_access_key_enabled = false` (storage),
`min_tls_version = "TLS1_2"`, `default_action = "Deny"` on storage
`network_rules` and vault `network_acls`.

Defaults: `artifacts_storage_account = "forecastmlartifacts"`,
`key_vault_name = "forecastmlvault"` (both
`^[a-z0-9]{3,24}$`, no hyphens).
`subscription_id` and `tenant_id` have **no defaults** — they are
required and must be supplied by the caller (variables only, never
hardcoded). `api_audience` (`api://forecast-ml-api`) is a separate
API audience and must differ from the workload-identity audience
`api://AzureADTokenExchange` (enforced by validation).

## No-residency-pin note

`location` (default `eastus2`) is a non-binding variable: it makes
**no residency or data-boundary claim** (D-SECRETS-PII-GDPR-1
§5.2(e)). No region literal in this stack implies a residency
commitment. The residency decision is taken at P2 IaC before P11;
artifacts created before that decision carry no residency
guarantee.

## Prerequisites

- Terraform `>= 1.3` (`terraform version`).
- Azure credentials from the environment only — set
  `ARM_SUBSCRIPTION_ID`, `ARM_TENANT_ID`, `ARM_CLIENT_ID`,
  `ARM_CLIENT_SECRET` or use the Azure CLI
  (`az login`). Never put credentials in `.tf`, `.tfvars`, or
  `conf/`.

## Apply order

Run from this directory:

```bash
terraform init
terraform plan -var="subscription_id=YOUR_SUBSCRIPTION_ID" -var="tenant_id=YOUR_TENANT_ID"
terraform apply -var="subscription_id=YOUR_SUBSCRIPTION_ID" -var="tenant_id=YOUR_TENANT_ID"
```

Always review `terraform plan` output before `terraform apply`.
Or export `TF_VAR_subscription_id` / `TF_VAR_tenant_id` and omit
`-var`. Tear down with `terraform destroy` (same `-var`).

## State

Local state by default (`terraform.tfstate`, gitignored — do not
commit it). For shared use, adopt the shipped
`azure/backend.tf.example` (azurerm) by copying it to `backend.tf`;
the pattern is commented in that file. Never commit
`*.tfstate*` or `*.tfvars` containing real values.

## Outputs → conf overlays

After apply, `terraform output` gives values to wire into the Kedro
env overlay:

- `artifacts_blob_url` (`https://forecastmlartifacts.blob.core.windows.net`)
  — consumed in `conf/azure/catalog.yml` via the single
  `${params:artifacts_blob_url}` interpolation (ONE mechanism, no
  direct-URL alternative). If you override
  `artifacts_storage_account`, set the matching
  `params:artifacts_blob_url` runtime param — not the filepaths.
- `key_vault_name` — Key Vault name for secret references.
- `aks_cluster_name` — AKS stub name (empty when `enable_aks` is
  false).
- `private_endpoint_ips` — storage + vault private endpoint IPs.
- `workload_identity_client_id` — managed identity client ID for
  workload identity binding.
- `oidc_issuer_url` — Entra v2.0 issuer URL
  (`https://login.microsoftonline.com/<tenant>/v2.0`).

## Encryption (CMEK) — production MUST set

Storage uses Microsoft-managed encryption by default (dev). That is
an accepted dev default only. **In production you MUST set a
customer-managed key:** `kms_key_id` (Key Vault key ID for storage
CMEK, `infra/azure/variables.tf`). Empty means managed encryption
and must not ship to production. Validation enforces it:
`production = true` with an empty `kms_key_id` fails
`terraform plan`. The Key Vault itself provides the CMEK key
material (Key Vault CMEK for storage when `var.kms_key_id` is
set).

## Validation

```bash
terraform fmt -check -recursive .
terraform init -backend=false -input=false
terraform validate
```
