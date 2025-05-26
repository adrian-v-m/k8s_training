#!/bin/bash

set -e

source .env

echo "Creating Kubernetes Secret 'hf-secrets' for HF_TOKEN..."
kubectl create secret generic hf-secrets \
    --from-literal=HF_TOKEN="$HF_TOKEN" \
    --dry-run=client -o yaml | kubectl apply -f -

echo "Kubernetes Secret 'hf-secrets' successfully created." 