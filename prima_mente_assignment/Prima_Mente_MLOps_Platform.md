# Large-Scale Training & Experimentation Platform
*32-node H200 cluster, NVLink + InfiniBand, and the main assumption is that it's an on-prem cluster*

## 1. System diagram

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#ffffff',
    'primaryTextColor': '#000000',
    'primaryBorderColor': '#000000',
    'lineColor': '#000000',
    'background': '#f8f9fa'
  }
}}%%

flowchart TD
    %% Control Plane
    subgraph CP["Kubeflow Control Plane"]
        KFP["Kubeflow Pipelines"]
        TO["Training Operator"]
        Kat["Katib HPO"]
    end

    %% Storage and Registry
    subgraph SR["Storage and Registry"]
        Ceph["Ceph File System<br/>shared snapshots"]
        NVMe["NVMe drive<br/>per-node cache"]
        GCS["Cloud Storage<br/>model files"]
        VReg[("Vertex AI<br/>Model Registry")]
    end

    %% Compute
    subgraph GC["GPU Cluster 32 nodes"]
        subgraph FCJ["Full-cluster Job"]
            L["32 training pods"]
        end
        subgraph PT["Parallel Trials"]
            Trials["Trial 1, Trial 2, ..., Trial N"]
        end
    end

    %% Tracking
    subgraph Track["Tracking"]
        Nep["Neptune.ai"]
    end

    %% Control-flow connections (solid arrows)
    KFP -->|submit PyTorchJob| TO
    KFP -->|start Katib Experiment| Kat
    Kat -->|create PyTorchJobs| TO
    TO -->|spawn & supervise| L
    TO -->|spawn & supervise| Trials

    %% Data-flow connections (dashed arrows)
    Ceph -.->|stage dataset at job start| NVMe
    NVMe -.->|flush checkpoints & logs| Ceph
    Trials <-.->|read data & write checkpoints| NVMe
    L <-.->|read data & write checkpoints| NVMe

    Trials -.->|stream metrics| Nep
    L -.->|stream metrics| Nep
    NVMe -.->|final checkpoint upload| GCS
    Ceph -.->|final checkpoint upload| GCS

    %% Static connections
    GCS -->|model file URIs| VReg

    %% Colorful styling for better readability
    style CP fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b
    style SR fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c
    style GC fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#1b5e20
    style PT fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
    style FCJ fill:#fff8e1,stroke:#f57f17,stroke-width:2px,color:#f57f17
    style Track fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#880e4f

    %% Node styling with contrasting text colors
    style KFP fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#0d47a1
    style TO fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#0d47a1
    style Kat fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#0d47a1
    
    style Ceph fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    style NVMe fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    style GCS fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    style VReg fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    
    style Trials fill:#ffcc80,stroke:#f57c00,stroke-width:2px,color:#e65100
    style L fill:#fff176,stroke:#f9a825,stroke-width:2px,color:#f57f17
    
    style Nep fill:#f8bbd9,stroke:#c2185b,stroke-width:2px,color:#880e4f

    %% Link styling for better contrast
    linkStyle default stroke:#2c3e50,stroke-width:3px
```

**Arrow Legend:**
- **Solid arrows (→)**: Control-flow connections - orchestration, job creation, and management commands
- **Dashed arrows (⇢)**: Data-flow connections - actual data movement, checkpoints, and metrics streaming

## 1.1. Hardware architecture

**Cluster Specifications (32 nodes, 256 GPUs total):**

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **GPUs** | 8× NVIDIA H200 per node with NVLink | Ultra-fast intra-node GPU communication |
| **CPUs** | 128-core Intel Sapphire Rapids per node | Data preprocessing and GPU feeding |
| **Memory** | 1.6 TB RAM per node | Massive dataset caching and preprocessing |
| **Local Storage** | 1 TB NVMe SSD per node | High-speed local data cache (7-14 GB/s) |
| **Interconnect** | InfiniBand for inter-node communication | NCCL-optimised for gradient synchronisation |


**Network Configuration:**
- NCCL with InfiniBand backend (`NCCL_IB_DISABLE=0`)

## 2. Component breakdown

| Layer | Technology | Role |
|-------|------------|------|
| **Workflow definition** | Kubeflow Pipelines (KFP) – an ML-oriented workflow engine that stores each step as a container and records lineage for every input and output in the form of a DAG. | A pipeline file (Python DSL → YAML) is version-controlled; One KFP pipeline per training job - only applies for multi-node training. |
| **Distributed job launcher** | Kubeflow Training Operator manages the PyTorchJob custom resource. It creates worker pods, configures PyTorch Distributed Data Parallel environment variables, and monitors job state. | Job specification lists replicas, CPU–GPU limits, restart policy, and elasticPolicy (see below). |
| **Elastic training** | Torch Elastic (part of PyTorch 2 runtime). Workers form a rendezvous group; the job continues if one or more workers drop out and can add new workers later. | `elasticPolicy {minReplicas:30, maxReplicas:32, maxRestarts:3}` in the PyTorchJob spec. |
| **Model sharding for very large networks** | FSDP (Fully Sharded Data Parallel) or DeepSpeed ZeRO-3. The choice is set by a command-line flag in the training container. | Parameters and optimiser state are partitioned across GPUs, allowing models that exceed single-GPU memory. |
| **Hyper-parameter search** | Katib – a Kubernetes service that creates an Experiment CRD (Custom Resource Definition). Algorithms (random search, Bayesian optimisation, CMA-ES) generate trial specs; each trial becomes a PyTorchJob. | Early-stopping policies compare intermediate metrics and terminate unpromising trials. |
| **Shared durable storage** | Ceph File System (CephFS) – a POSIX-compliant clustered file system. Data blocks are triple-replicated over separate Ceph OSD servers; snapshots are copy-on-write and completed in milliseconds. | CephFS mounts into every pod through a PersistentVolumeClaim with ReadWriteMany mode. |
| **Per-node cache** | NVMe SSD inside each node, exposed as a local Persistent Volume that only pods on that server can mount. | An init-container copies data from CephFS to NVMe if the required hash marker is absent; subsequent reads are local. |
| **Experiment logging** | Neptune.ai – client library initialises a run object; every metric, parameter, and small artefact is streamed to Neptune's back-end. Web UI scales to thousands of runs. | Sensitive API tokens are supplied via Kubernetes Secrets. |
| **Model registry** | Vertex AI Model Registry – each trained model is registered as a Model; file payload is stored in GCS. The registry records version, metadata, and lineage to the training pipeline. | Vertex AI client SDK is called in a pipeline step. |
| **Container images** | Artifact Registry – private Docker registry. CI pipeline pushes images and writes the digest into the pipeline YAML to ensure immutability. | Vulnerability scan and signature validation are enabled on every push. |
| **Monitoring** | Prometheus scrapes node exporters and NVIDIA DCGM exporter; Grafana displays dashboards. | Alertmanager sends alerts if GPU temperature exceeds limit or Ceph latency > 5 ms. |

## 2.1. Advanced data management strategy

**Two-Tier Storage Architecture:**

1. **Durable Storage (CephFS)**
   - Triple-replicated data blocks across separate OSD servers
   - Copy-on-write snapshots completed in milliseconds
   - POSIX-compliant for seamless application integration
   - Snapshot-based experiment versioning for reproducibility

2. **High-Speed Cache (NVMe)**
   - 1 TB per node for ultra-fast local access
   - Automatic data staging with hash-based validation
   - Background synchronisation to durable storage
   - Local Persistent Volumes with node affinity

**Data Staging Process:**
```bash
# Hash-based caching mechanism
DATASET_HASH=$(sha256sum /ceph/datasets/${DATASET_ID}/manifest.json | cut -d' ' -f1)
if [ ! -f /nvme/.cache_${DATASET_HASH} ]; then
  echo "Staging dataset ${DATASET_ID} to local NVMe..."
  rsync -av --progress /ceph/datasets/${DATASET_ID}/ /nvme/data/
  touch /nvme/.cache_${DATASET_HASH}
  echo "Dataset staged successfully"
else
  echo "Dataset already cached locally"
fi
```

## 3. Execution plan

### A. Multiple small parallel experiments (Katib)

KFP pipelines are not necessary since these are single-node experiments where there is no need to containerise the steps of the training process. Instead, a single script that handles data pre-processing and model training is containerised and used to run in a training job.

1. **Start** - AI researcher selects search space in the KFP UI and triggers HPO (Hyperparameter Optimisation) pipeline.

2. **Katib generates dozens of trials**; each trial is a PyTorchJob requesting 1–8 GPUs with gang scheduling to ensure atomic resource allocation.

3. **Data copy via trial init container**:
   ```bash
   if [ ! -f /nvme/.${SHA} ]; then
     rsync -a /ceph/datasets/${SHA}/ /nvme/ && touch /nvme/.${SHA}
   fi
   ```

4. **Training** - Script runs on local NVMe data. Logs `loss`, `perplexity`, `tokens/second` to Neptune.ai. Checkpoints go to `/nvme/ckpt` (NVMe PV).

5. **Background sync** - Side-car container copies new checkpoints in an async way to CephFS every 10 min, then to `gs://checkpoints/<run-id>/` (GCS).

6. **Fault Tolerance** - Independent trials minimise fault impact; failed trials are automatically retried by Katib with configurable `maxFailedTrialCount`.

7. **Completion** - Katib returns best hyperparameter set; pipeline can optionally retrain the best configuration on more GPUs.

### B. Single large-scale job (32 nodes, 256 GPU)

KFP pipeline is fully implemented for the multi-node PyTorch job. This training job uses the best hyperparameter set obtained in the previous step.

1. **Start** - KFP Pre-train Pipeline called with model config and dataset snapshot ID.

2. **Resource Reservation** - PyTorchJob with high PriorityClass can preempt lower-priority experiments to claim entire cluster.

3. **Data Parallelisation** - Data-prep job shards data across 32 NVMe PVs (InfiniBand copy ≈ 2 min for 2 TB).

4. **Distributed Setup** - Training Operator configures distributed environment:
   ```yaml
   env:
   - name: MASTER_ADDR
     value: "ftjb-master-0"
   - name: MASTER_PORT
     value: "23456"
   - name: WORLD_SIZE
     value: "256"
   - name: NCCL_IB_DISABLE
     value: "0"
   - name: NCCL_DEBUG
     value: "INFO"
   ```

5. **Training** - `PyTorchJob` specifies `replicas=32`. Each worker starts with `torchrun --nnodes=$WORLD_SIZE --nproc-per-node=8 train.py --fsdp`. Training metrics are logged to Neptune.ai.

6. **Fault handling** - TorchElastic continues with reduced workers; Kubernetes reschedules failed pods; training resumes at next rendezvous barrier.

7. **Checkpoint Strategy** - Multi-layer checkpointing:
   - **Fast checkpoints**: Every 15 min to local NVMe
   - **Durable checkpoints**: Every 30 min to CephFS
   - **Backup checkpoints**: Every 2 hours to GCS
   - **Checkpoint validation**: Hash verification to prevent corruption

8. **Evaluation** - Separate container runs for the validation script; evaluation metrics streamed to Neptune.ai.

9. **Model registration**: Vertex AI call: `model.upload_version(artifact_uri="gs://pretrain/ckpt/final/", dataset ID, Git SHA, Neptune experiment ID)`

10. **Container registration**: the pipeline containers which produce the data and model (weights, artifaces, hyperparameters, etc.) are registered in the GCP Artifact Registry.

11. **Cleanup**: job PVC and NVMe cache cleaned; cluster autoscaler may power down idle nodes.

## 3.1. Advanced fault tolerance mechanisms

**TorchElastic Configuration:**
```yaml
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 30
    maxReplicas: 32
    maxRestarts: 3
    rdzvConfigs:
    - key: timeout
      value: "900"
```

**Recovery Strategies:**
1. **Node failure**: TorchElastic continues with remaining nodes
2. **Pod failure**: Kubernetes reschedules; job waits at rendezvous
3. **Network partition**: NCCL timeout triggers re-initialisation
4. **Checkpoint corruption**: Automatic fallback to previous checkpoint
5. **Cluster maintenance**: Graceful shutdown with `terminationGracePeriodSeconds: 300`

## 4. Justification for selected tools

- **Kubeflow Pipelines** – provides a reproducible, versioned description of ML workflows and stores metadata and artefacts for every step.
- **Training Operator + PyTorchJob** – encapsulates best practices for distributed PyTorch on Kubernetes, including elastic policies and automatic rendezvous.
- **Torch Elastic** – allows large jobs to survive node loss without restart, a critical requirement for multi-day pre-training.
- **FSDP / DeepSpeed** – enable models that exceed single-GPU memory by partitioning parameters and optimiser state.
- **CephFS** – offers POSIX semantics, replication, and instant snapshots; suitable as a single, durable store for datasets and checkpoints.
- **NVMe cache** – eliminates repeated network reads, delivering 7–14 GB/s per node while keeping the design simple (copy-on-read, background flush).
- **Katib** – integrates with Kubeflow and supports advanced search algorithms; trials are native Kubernetes jobs, not a separate service.
- **Neptune.ai** – captures metrics, parameters, and small files with minimal code; the back-end handles thousands of concurrent runs.
- **Vertex AI Model Registry + GCS** – separates heavy binary storage (Cloud Storage) from catalogue metadata (Registry), providing access control and deployment history.
- **Artifact Registry** – central, signed image store via CI; every pipeline component image is immutable and scanned for vulnerabilities.
- **Monitoring** - NVIDIA DCGM exporter measures GPU errors, clock throttling, memory use. Ceph exporter measures client latency and queue length. Prometheus thresholds trigger alerts. All logs and metrics are visualised with Grafana. Custom dashboards for training progress.

## 5. Performance optimisations

### 5.1. Training optimisations

| Optimisation | Implementation | Expected Gain |
|-------------|----------------|---------------|
| **Mixed Precision** | PyTorch AMP with FP16/BF16 | 2x training throughput |
| **Gradient Compression** | NCCL with compression algorithms | 30-50% communication reduction |
| **Data Loading** | Multi-process DataLoader with `num_workers=8` | Eliminate I/O bottlenecks |
| **Memory Prefetching** | GPU memory prefetching for next batch | Overlap compute and data transfer |
| **NUMA Optimisation** | CPU affinity binding to local memory | Minimise memory access latency |
| **GPUDirect Storage** | Direct GPU-to-NVMe data path | Bypass CPU for data reads |

### 5.2. Network optimisations

```yaml
# NCCL Environment Variables
env:
- name: NCCL_IB_DISABLE
  value: "0"
- name: NCCL_IB_HCA
  value: "mlx5_0,mlx5_1"
- name: NCCL_IB_GID_INDEX
  value: "3"
- name: NCCL_NET_GDR_LEVEL
  value: "2"
- name: NCCL_NET_GDR_READ
  value: "1"
```

## 6. Security framework

### 6.1. Multi-Layer security

| Layer | Implementation | Purpose |
|-------|----------------|---------|
| **Authentication** | Kubeflow + Istio + Google IAP | User identity verification |
| **Authorisation** | Kubernetes RBAC + Istio AuthPolicy | Resource access control |
| **Network Security** | Istio service mesh + NetworkPolicies | Traffic encryption and isolation |
| **Container Security** | minimal images | Container runtime isolation |
| **Data Security** | Encryption at rest + in transit | Data protection |
| **Secret Management** | Kubernetes Secrets + rotation | Secure credential handling |

### 6.2. Network policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: training-pods-policy
spec:
  podSelector:
    matchLabels:
      app: pytorch-training
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: ceph-system
  - to:
    ports:
    - protocol: TCP
      port: 443  # Neptune.ai HTTPS
  - to:
    ports:
    - protocol: TCP
      port: 443  # GCS HTTPS
```

## 7. Comprehensive monitoring

### 7.1. Infrastructure monitoring

**Prometheus Metrics:**
- **GPU Metrics**: Utilisation, memory, temperature, power consumption
- **Network Metrics**: InfiniBand throughput, packet loss, latency
- **Storage Metrics**: CephFS latency, IOPS, throughput
- **Node Metrics**: CPU, memory, disk utilisation

**NVIDIA DCGM Exporter Configuration:**
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    spec:
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.8-3.1.5-ubuntu20.04
        securityContext:
          capabilities:
            add: ["SYS_ADMIN"]
        volumeMounts:
        - name: proc
          mountPath: /hostproc
        env:
        - name: DCGM_EXPORTER_COLLECTORS
          value: "/etc/dcgm-exporter/dcp-metrics-included.csv"
```


## 8. Additional considerations

| Topic | Implementation |
|-------|----------------|
| **CI/CD** | GitHub Actions build container, push to Artifact Registry, compile pipeline (`kfp dsl compile`), and upload new version. Image tag includes Git SHA for reproducibility. Container vulnerability scanning with Trivy. |
| **Security** | Kubernetes RBAC isolates teams; Istio side-car enforces egress rules so training pods can reach only Ceph, Neptune and Cloud Storage. Secrets (tokens, keys) are mounted read-only. |
| **Monitoring** | NVIDIA DCGM exporter measures GPU errors, clock throttling, memory use. Ceph exporter measures client latency and OSD queue length. Prometheus thresholds trigger alerts. All logs and metrics are visualised with Grafana. Custom dashboards for training progress. |
| **Scalability** | Cluster can expand to 64 nodes by updating the `replicas` field. If Ceph cannot meet read bandwidth (including if per-node read > 25-30 GB/s), a future upgrade to DAOS or Lustre is possible without pipeline changes. Horizontal Pod Autoscaler for supporting services. |
| **Cost** | Torch Elastic permits use of spot GPU nodes; checkpoints every 15 min to avoid > 15 min loss. Weekly CronJob cleans NVMe caches older than 7 days. Cluster autoscaling with node auto-provisioning and preemptible instances. |

### 8.1. Cost optimisation strategies

**Automatic Cleanup CronJob:**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nvme-cleanup
spec:
  schedule: "0 2 * * 0"  # Weekly at 2 AM Sunday
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cleanup
            image: busybox
            command:
            - /bin/sh
            - -c
            - find /nvme -type f -mtime +7 -delete
```

### Example Kubeflow pipeline DAG

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#ffffff',
    'primaryTextColor': '#000000',
    'primaryBorderColor': '#000000',
    'lineColor': '#000000',
    'background': '#f8f9fa'
  }
}}%%

flowchart TD
  A["🗂️ Data preparation<br/>dataset staging"] --> B["🔤 Tokenise and pack<br/>text preprocessing"]
  B --> C["📊 Data sharding<br/>32 NVMe PVs<br/>distributed storage"]
  C --> D["🚀 Train model<br/>PyTorchJob<br/>distributed training"]
  D --> E["📈 Evaluate<br/>model validation<br/>performance metrics"]
  E --> F["📦 Register in Vertex AI<br/>model versioning<br/>deployment ready"]
  D -->|stream metrics| N["🌊 Neptune.ai<br/>experiment tracking"]
  D -->|save checkpoints| G["💾 CephFS + GCS backup<br/>fault tolerance"]
  F --> H["🧹 Cleanup resources<br/>cost optimization"]

  %% Colorful styling for pipeline stages
  style A fill:#e3f2fd,stroke:#1976d2,stroke-width:3px,color:#0d47a1
  style B fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,color:#4a148c
  style C fill:#e8f5e8,stroke:#388e3c,stroke-width:3px,color:#1b5e20
  style D fill:#fff3e0,stroke:#f57c00,stroke-width:3px,color:#e65100
  style E fill:#fce4ec,stroke:#c2185b,stroke-width:3px,color:#880e4f
  style F fill:#e0f2f1,stroke:#00796b,stroke-width:3px,color:#004d40
  style N fill:#e1f5fe,stroke:#0288d1,stroke-width:3px,color:#01579b
  style G fill:#f1f8e9,stroke:#689f38,stroke-width:3px,color:#33691e
  style H fill:#fff8e1,stroke:#ffa000,stroke-width:3px,color:#ff6f00

  %% Link styling for better contrast
  linkStyle default stroke:#37474f,stroke-width:3px
  linkStyle 5 stroke:#0288d1,stroke-width:3px,stroke-dasharray:5
  linkStyle 6 stroke:#689f38,stroke-width:3px,stroke-dasharray:5
```

**Pipeline Flow Legend:**
- **Solid arrows (→)**: Sequential pipeline steps - main execution flow from data preparation to cleanup
- **Dashed arrows (⇢)**: Parallel outputs - concurrent logging and backup operations during training


## Future potential improvements

### 1. Pipeline Parallelism
- **Function**: Splits a large model into sequential stages so each GPU processes different layers simultaneously, rather than replicating the entire model on every GPU.  
- **Integration**:  
  1. Add `--pipeline-parallel K` to the PyTorch training entry point.  
  2. Existing `PyTorchJob` pods form stages.  
  3. Torch Elastic handles dropped or added stage pods.  
- **Value**:  
  - Enables significantly larger models within the same 8 GPUs/node configuration.  
  - Raises overall GPU usage by ~10 % compared to pure data parallel.  
  - No changes to storage or orchestration required.

### 2. KServe for Inference
- **Function**: Deploys trained models as Kubernetes “InferenceService” objects that automatically scale GPUs to zero when idle and roll out new versions via traffic-splitting.  
- **Integration**:  
  1. After the final checkpoint lands in Cloud Storage, add a KServe manifest pointing to that URI.  
  2. Knative autoscaling spins GPU pods up or down based on request volume.  
- **Value**:  
  - Eliminates manual inference serving.  
  - Reuses the same container images from training.  
  - Ensures zero-cost idle time.  
  - Enables safe canary releases for new model versions.

### 3. Feature Store (Feast)
- **Function**: Version-tags every derived feature once (e.g., token counts, positional bins) in Parquet for offline and a key-value store for online; serves identical data to training and serving.  
- **Integration**:  
  1. Add a nightly `feast pushing` job that writes a snapshot to CephFS.  
  2. Each trial or full-cluster run reads features from the cached snapshot on NVMe instead of recomputing.  
- **Value**:  
  - Reduces preprocessing significantly per trial since onlt copying NVMe files is needed. 
  - Prevents drift between training and serving features.  
  - Cuts redundant compute across hundreds of experiments.
