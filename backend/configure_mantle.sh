#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${BACKEND_DIR}/.env"
TEMP_ENV="$(mktemp "${BACKEND_DIR}/.env.XXXXXX")"

cleanup() {
  unset BEDROCK_KEY
  if [[ -f "${TEMP_ENV}" ]]; then
    rm -f -- "${TEMP_ENV}"
  fi
}
trap cleanup EXIT

printf "Paste your Amazon Bedrock API key (input is hidden): "
IFS= read -r -s BEDROCK_KEY
printf "\n"

# Accept either the bare key or a command copied from AWS documentation.
BEDROCK_KEY="${BEDROCK_KEY#export AWS_BEARER_TOKEN_BEDROCK=}"
BEDROCK_KEY="${BEDROCK_KEY#AWS_BEARER_TOKEN_BEDROCK=}"
BEDROCK_KEY="${BEDROCK_KEY#Bearer }"

if [[ -z "${BEDROCK_KEY}" ]]; then
  printf "No key was entered; backend/.env was not changed.\n" >&2
  exit 1
fi

umask 077
if [[ -f "${ENV_FILE}" ]]; then
  grep -Ev '^(AWS_BEARER_TOKEN_BEDROCK|AWS_REGION|OPENAI_BASE_URL|MANTLE_MODEL|MANTLE_MAX_OUTPUT_TOKENS)=' "${ENV_FILE}" > "${TEMP_ENV}" || true
fi

{
  printf "\n# Amazon Bedrock Mantle (backend only)\n"
  printf "AWS_BEARER_TOKEN_BEDROCK=%s\n" "${BEDROCK_KEY}"
  printf "AWS_REGION=us-east-1\n"
  printf "OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1\n"
  printf "MANTLE_MODEL=qwen.qwen3-next-80b-a3b-instruct\n"
  printf "MANTLE_MAX_OUTPUT_TOKENS=1100\n"
} >> "${TEMP_ENV}"

mv -- "${TEMP_ENV}" "${ENV_FILE}"
chmod 600 "${ENV_FILE}"
printf "Saved Bedrock Mantle settings to backend/.env. The API key was not displayed.\n"
