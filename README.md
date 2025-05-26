# Kubernetes PyTorchJob trainer

Everything needed to run distributed fine-tuning jobs using Kubeflow PyTorchJob.

## What's in here

- `Dockerfile` - builds the container with training components
- `fine_tune_job.yaml` - tells Kubernetes how to run the training job (`MODEL_URI` and `DATASET_REPO_ID` are hardcoded)
- `model_train_lora.py` - main training script with LoRA fine-tuning
- `graceful_shutdown_callback.py` - handles graceful shutdowns
- `setup-env.sh` - creates a Kubernetes secret for `HF_TOKEN`
- `.env` - Hugging Face token (gitignored)
- `.gitignore` - keeps sensitive files out of git

## Getting started

Connect to cluster, build a Docker image, deploy the job.

### Connecting to the cluster

A `kubeconfig` file (usually `kubeConfig.yaml`) with connection details for the cluster is needed.

```bash
export KUBECONFIG=kubeConfig.yaml
```

### Building the training image

Build a Docker image with the training code:

```bash
# Make sure Docker is running
docker buildx build --load -t quay.io/oriedge/trainer-huggingface:test trainer/
```

For non-local clusters, push this image to a container registry the cluster can access.

### Setting up the Hugging Face token

The `.env` file in `trainer/` contains the Hugging Face token (gitignored).

Run the setup script:

```bash
cd trainer/
./setup-env.sh
cd ..
```

This creates a Kubernetes secret called `hf-secrets` with the token.

### Deploying the job

Deploy the training job:

```bash
kubectl apply -f trainer/fine_tune_job.yaml
```

The job automatically downloads the model and dataset using the token.

## Running distributed training (manual way)

**Note**: This is the manual way to run training inside containers. Great for experimenting, debugging, or when control over each container is needed. In production, this process would be automated.

The `fine_tune_job.yaml` sets up a PyTorchJob with one master and one worker. When the job starts, Kubeflow sets up environment variables like `MASTER_ADDR`, `MASTER_PORT`, `RANK`, and `WORLD_SIZE` in each pod.

Once pods are running, manually start training using `kubectl exec`:

**1. Copy the updated script (if changed):**

```bash
export KUBECONFIG=kubeConfig.yaml

kubectl cp trainer/model_train_lora.py ftjb-master-0:/app/model_train_lora.py -c pytorch
kubectl cp trainer/model_train_lora.py ftjb-worker-0:/app/model_train_lora.py -c pytorch
```

**2. Kill old training processes:**

```bash
kubectl exec ftjb-master-0 -c pytorch -- pkill -f torchrun || true
kubectl exec ftjb-worker-0 -c pytorch -- pkill -f torchrun || true
```

**3. Start training on master (RANK 0):**

```bash
kubectl exec ftjb-master-0 -- bash -c "cd /app && MASTER_ADDR=ftjb-master-0 MASTER_PORT=23456 nohup torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr=ftjb-master-0 --master_port=23456 model_train_lora.py --model_name /workspace/model/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 --dataset_path /workspace/dataset --output_dir /workspace/output --local_files_only --add_special_tokens > /app/master_training.log 2>&1 &"
```

**4. Start training on worker (RANK 1):**

```bash
kubectl exec ftjb-worker-0 -- bash -c "cd /app && MASTER_ADDR=ftjb-master-0 MASTER_PORT=23456 nohup torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 --master_addr=ftjb-master-0 --master_port=23456 model_train_lora.py --model_name /workspace/model/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 --dataset_path /workspace/dataset --output_dir /workspace/output --local_files_only --add_special_tokens > /app/worker_training.log 2>&1 &"
```

**Parameters:**

- `--nnodes=2`: 2 nodes (pods) - one master, one worker
- `--nproc_per_node=1`: Each node runs 1 training process
- Script arguments like `--model_name` point to where containers downloaded data
- Logs go to `/app/master_training.log` and `/app/worker_training.log`

The master pod is the meeting point, worker finds it through Kubernetes networking. They sync up and train in parallel.

**Note**: This is a manual approach and it's for experimenting and having more control of the training in each container. In production, this would be automated.

## Configuration

- **`HF_TOKEN`**: Hugging Face token
  - lives in `trainer/.env`
  - loaded into Kubernetes as secret `hf-secrets` by `trainer/setup-env.sh`
  - used by containers to download models/datasets
- **`MODEL_URI` & `DATASET_REPO_ID`**: hardcoded in `trainer/fine_tune_job.yaml`

## Security

- `HF_TOKEN` stays in local `trainer/.env` (gitignored) and as Kubernetes secret
- `trainer/setup-env.sh` handles secure transfer to cluster

## Monitoring

Check job status:

```bash
kubectl get pytorchjobs
kubectl describe pytorchjob ftjb

# Check training logs
kubectl exec ftjb-master-0 -c pytorch -- cat /app/master_training.log
kubectl exec ftjb-worker-0 -c pytorch -- cat /app/worker_training.log

# Watch pod logs directly
kubectl logs -f ftjb-master-0 -c pytorch
kubectl logs -f ftjb-worker-0 -c pytorch
``` 