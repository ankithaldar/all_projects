# AWS secret reference for forecast-ml-framework (P2-S1).
#
# Shell only: this file creates the secret shell (name only).
# No value is stored here and no value-bearing resource is
# declared. Set the value out-of-band with the AWS CLI
# after apply, for example:
#   aws secretsmanager put-secret-value --secret-id <name> --secret-string <value>
# The secret name is a variable (default below) so each
# environment can point at its own shell. Auth comes from
# the environment only (AWS_PROFILE or AWS_* env vars);
# nothing is hardcoded here.

variable "secret_name" {
  description = "Name of the Secrets Manager secret shell. The value is set out-of-band via the AWS CLI, never in Terraform."
  type        = string
  default     = "forecast-ml-api-secrets"
}

resource "aws_secretsmanager_secret" "main" {
  name        = var.secret_name
  description = "forecast-ml-framework API secret shell (value set out-of-band)."
  tags        = var.tags
}

output "secret_id" {
  description = "ID of the Secrets Manager secret shell (name reference only, never a value)."
  value       = aws_secretsmanager_secret.main.id
}
