# GCP secret reference for forecast-ml-framework (P2-S1).
#
# Shell only: this file creates the secret shell (name only)
# and grants the dedicated workload service account read
# access. No value is stored here and no value-bearing
# resource is declared. Set the value out-of-band with the
# gcloud CLI after apply, for example:
#   gcloud secrets versions add <secret> --data-file=-
# The secret name is a variable (default below) so each
# environment can point at its own shell. Auth comes from
# the environment only (GOOGLE_APPLICATION_CREDENTIALS);
# nothing is hardcoded here.

variable "secret_name" {
  description = "Name of the Secret Manager secret shell. The value is set out-of-band via the gcloud CLI, never in Terraform."
  type        = string
  default     = "forecast-ml-api-secrets"
}

resource "google_secret_manager_secret" "main" {
  secret_id = var.secret_name

  replication {
    auto {}
  }

  labels = var.labels
}

resource "google_secret_manager_secret_iam_member" "workloads" {
  secret_id = google_secret_manager_secret.main.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.workloads.email}"
}

output "secret_id" {
  description = "ID of the Secret Manager secret shell (name reference only, never a value)."
  value       = google_secret_manager_secret.main.id
}
