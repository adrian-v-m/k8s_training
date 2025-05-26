# Kubernetes PyTorchJob Trainer

This directory contains the configuration and scripts for running distributed fine-tuning jobs using Kubeflow PyTorchJob.

## Files

- `Dockerfile` - Container definition for the training environment
- `fine_tune_job.yaml` - Kubernetes PyTorchJob configuration
- `model_train_lora.py` - Main training script with LoRA fine-tuning
- `graceful_shutdown_callback.py` - Callback for handling graceful shutdowns
- `setup-env.sh` - Script to create a Kubernetes Secret for `HF_TOKEN` from the `.env` file
- `.env` - Environment variables file for `HF_TOKEN` (create this locally)
- `.gitignore` - Specifies intentionally untracked files that Git should ignore (e.g., `.env`)

## Setup

### 1. Create Environment Variables File (`.env`)

Create a `.env` file in this directory (`k8s_training/trainer/.env`) with your actual Hugging Face token:

```bash
# Hugging Face Configuration
HF_TOKEN=your_actual_huggingface_token_here
```

**Important**: The `.env` file is ignored by git (as defined in `.gitignore`) to prevent committing sensitive information.

### 2. Setup Kubernetes Secret for `HF_TOKEN`

Run the setup script in this directory to create the necessary Kubernetes Secret. Ensure `kubectl` is configured to point to your target cluster.

```bash
./setup-env.sh
```

This script will:
- Read the `HF_TOKEN` from your `.env` file
- Create (or update) a Kubernetes Secret named `hf-secrets` containing the `HF_TOKEN`

### 3. Deploy the Training Job

Deploy the PyTorchJob to your Kubernetes cluster:

```bash
kubectl apply -f fine_tune_job.yaml
```

The `initContainers` in the job will use the `HF_TOKEN` from the `hf-secrets` Secret for downloading the model and dataset.

## Environment Variables & Configuration

- **`HF_TOKEN`**: Hugging Face access token
    - Managed via the local `.env` file
    - Loaded into Kubernetes as a Secret named `hf-secrets` by `setup-env.sh`
    - Injected into `initContainers` as an environment variable `HF_TOKEN`
    - Referenced in `initContainer` arguments as `$(HF_TOKEN)`
- **`MODEL_URI` & `DATASET_REPO_ID`**: These are currently hardcoded directly in the `fine_tune_job.yaml` within the `initContainer` arguments. If you need to make these configurable, you could extend the `.env` and `setup-env.sh` script to manage them via a ConfigMap, similar to how `HF_TOKEN` is handled with a Secret.

## Security

- The `HF_TOKEN` is stored in a local `.env` file (git-ignored) and as a Kubernetes Secret
- The `setup-env.sh` script facilitates the secure transfer of the `HF_TOKEN` to the Kubernetes cluster

## Monitoring

Monitor the job status with:

```bash
kubectl get pytorchjobs
kubectl describe pytorchjob ftjb
kubectl logs -f ftjb-master-0 # Or ftjb-worker-0
``` 