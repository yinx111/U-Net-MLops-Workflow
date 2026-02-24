# U-Net++ Satellite Segmentation MLOps Workflow

This repository provides an end-to-end MLOps workflow for multispectral satellite image segmentation based on U-Net++.

It combines:

- Model training/evaluation/inference scripts (PyTorch)
- DVC data + artifact versioning
- MLflow tracking and model registry integration
- FastAPI web inference app
- GitHub Actions CI and CD pipelines
- DagsHub-backed DVC/MLflow remote setup

## Features

- Reproducible pipeline via `dvc.yaml`
- Quality gate before model registration
- MLflow run tracking under a single experiment
- CI checks with lint/type-check, smoke tests, and mini DVC validation
- Dockerized web app deployment
- CD model delivery from DVC remote to deployment host
- CD guardrail: deployment on `push` only runs when `dvc.lock` changes

## Repository Layout

```text
.
├── app/                          # FastAPI app + UI
├── configs/train.yaml            # Training/MLflow/gate config
├── deploy/docker-compose.deploy.yml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── check_metrics.py
│   └── register_model.py
├── dvc.yaml                      # Pipeline definition
├── dvc.lock                      # Pipeline lock (deployment-critical)
├── dataset_split.dvc             # DVC dataset pointer
├── docker-compose.yml            # Local app compose
├── requirements.txt              # Local training/runtime deps
├── requirements_CI.txt           # CI deps
├── requirements_app.txt          # App image deps
└── .github/workflows/
    ├── ci.yml
    └── cd-app.yml
```

## Prerequisites

- Python 3.9
- Linux + CUDA recommended for local GPU training
- DVC remote and MLflow tracking server (DagsHub in current setup)

## Installation

```bash
conda create --name smp python=3.9 -y
conda activate smp

# Optional: install your CUDA-matching PyTorch first
# Example (CUDA 11.3):
# pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```


## Configure DVC Remote (DagsHub)

```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl <your_dagshub_repo_url>.s3
dvc remote modify origin --local access_key_id <your_token>
dvc remote modify origin --local secret_access_key <your_token>
```

Use your DagsHub access token for both key and secret in this setup.

<img width="460" height="473" alt="dvc remote setting ok" src="https://github.com/user-attachments/assets/2249e74f-2f90-434d-85dc-86f5d74ad967" />


## Configure MLflow

Create a local `.env` (or copy from `.env.example`) and set:

```env
MLFLOW_TRACKING_URI=<your_dagshub_repo_url>.mlflow
MLFLOW_TRACKING_USERNAME=<your_username>
MLFLOW_TRACKING_PASSWORD=<your_token>
```

<img width="1147" height="503" alt="image" src="https://github.com/user-attachments/assets/3ec18948-ec86-4aa1-99b2-c8350c7a7b8f" />


Recommended local inference-related values in `.env`:

```env
CHECKPOINT_PATH=./outputs/model.pth
IMG_PATH=./test_img/area_test1.tif
OUT_DIR=./outputs/infer
FORCE_CPU=0
CUDA_VISIBLE_DEVICES=0
```

Do not use deployment container paths (such as `/workspace/...`) in local training `.env`.

## DVC Pipeline

Defined in `dvc.yaml`:

1. `train` -> `outputs/model.pth`, `outputs/metrics.json`
2. `evaluate` -> `outputs/eval/*`
3. `inference` -> `outputs/infer/*`
4. `gate` -> `outputs/gate.ok`
5. `register` -> `outputs/register.ok`

Common commands:

```bash
# Full pipeline
dvc repro

# Typical model release stages
dvc repro train evaluate inference gate register

# Local status
dvc status

# Cloud sync status
dvc status -c
```

## MLflow Behavior

- Config source: `configs/train.yaml` (`mlflow` section)
- Default experiment: `u-net-workflow`
- New training runs are appended as runs under one experiment
- If the experiment is soft-deleted, training restores and reuses the same experiment name

<img width="1913" height="913" alt="image" src="https://github.com/user-attachments/assets/5a4baa73-78dd-4b63-832b-46810e76dfe6" />


## Local Web App

For local app usage:

```bash
docker compose up -d --build
```

Endpoints:

- UI: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/healthz`

`/healthz` validates checkpoint existence/readability.

<img width="699" height="333" alt="image" src="https://github.com/user-attachments/assets/3d9f9bea-7b2d-456e-ac62-9b1aaa63ee80" />


## CI (GitHub Actions)

Workflow: `.github/workflows/ci.yml`

<img width="983" height="430" alt="image" src="https://github.com/user-attachments/assets/cf6686e6-9d86-496f-ab20-9bc77f8ace73" />


### Triggers

- `push` to `dev`, `main`, `master`
- `pull_request` targeting `dev`, `main`, `master`

### Jobs

1. `lint-and-typecheck`
- Ruff
- MyPy (lenient config)

2. `tests`
- Installs `requirements_CI.txt`
- Runs `pytest -q`

3. `dvc-mini`
- Installs `requirements_CI.txt`
- Copies `dataset_mini` to `dataset_split`
- Runs a lightweight DVC pipeline on CPU:
  - `dvc repro --force evaluate inference`

## CD (GitHub Actions)

Workflow: `.github/workflows/cd-app.yml`

### Triggers

- `push` to `dev`, `main`, `master`
- `workflow_dispatch` (manual)

### Deploy Guardrail

On `push`, build/deploy runs only when `dvc.lock` changed.

### Deployment Flow

1. Build and push app image to GHCR
2. Pull model artifact from DVC remote on the GitHub runner
3. Upload compose file + model to remote host via SCP
4. Run remote `docker compose up -d`
5. Wait for health check; rollback to previous image on failure

### Model Path Mapping

- Remote host file:
  - `<STAGING_DEPLOY_PATH>/deploy/models/<GITHUB_SHA>/model.pth`
- Container path:
  - `/workspace/models/<GITHUB_SHA>/model.pth`

Deployment compose file:

- `deploy/docker-compose.deploy.yml`

<img width="937" height="354" alt="image" src="https://github.com/user-attachments/assets/0805b789-28e8-40c7-988f-360c3bbd345f" />


## GitHub Environment (`staging`) Setup

### Secrets

| Name | Description | Example |
|---|---|---|
| `STAGING_SSH_HOST` | Deployment server public IP or DNS. | `1.2.3.4` |
| `STAGING_SSH_USER` | SSH login user. | `ubuntu` |
| `STAGING_SSH_KEY` | Private SSH key for the user. | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `STAGING_DEPLOY_PATH` | Absolute remote deployment directory. | `/home/ubuntu/webapp_deploy` |
| `STAGING_GHCR_USERNAME` | GHCR pull username on remote host. | `your-github-user` |
| `STAGING_GHCR_TOKEN` | GHCR token/PAT with `read:packages`. | `ghp_xxx...` |
| `STAGING_DVC_ACCESS_KEY_ID` | DVC remote access key (S3-compatible). | `AKIA...` |
| `STAGING_DVC_SECRET_ACCESS_KEY` | DVC remote secret key. | `xxxx` |
| `CLOUDFLARED_TOKEN` | Cloudflare tunnel token. | `eyJ...` |

### Variables

| Name | Description | Default / Typical |
|---|---|---|
| `APP_HOST` | FastAPI bind host in container. | `0.0.0.0` |
| `APP_PORT` | Exposed app port. | `8000` |
| `MAX_UPLOAD_MB` | Upload size limit. | `200` |
| `INFERENCE_TIMEOUT_SECONDS` | Inference timeout. | `3600` |
| `FORCE_CPU` | Force CPU inference (`1/0`). | `1` |
| `CUDA_VISIBLE_DEVICES` | Visible GPUs for inference container. | empty |
| `MODEL_DVC_PATH` | DVC model path pulled in CD. | `outputs/model.pth` |
| `MODEL_SHA256` | Optional checksum pin. | empty |
| `DVC_AWS_REGION` | Region for DVC S3 client. | `us-east-1` |


<img width="940" height="853" alt="image" src="https://github.com/user-attachments/assets/cff7bc62-5978-49cd-9196-227cb72c7514" />


## Recommended Release Sequence

```bash
# 1) Train and update outputs
dvc repro (dvc repro --force)

# 2) Push artifacts to DVC remote
dvc push outputs/model.pth
# or: dvc push

# 3) Commit pipeline lock/code and push
git add dvc.lock
git commit -m "update model"
git push
```

CD pulls the model referenced by the pushed commit's `dvc.lock`, not simply the newest remote object.

## Troubleshooting

1. `FileNotFoundError: /workspace/outputs/model.pth` in local runs
- Cause: local `.env` accidentally uses deployment container path
- Fix: set `CHECKPOINT_PATH=./outputs/model.pth`

2. `dvc status -c` fails due to missing `dvc_s3`
- Fix: `pip install "dvc[s3]==3.66.1"`

3. CD SCP step fails with `dial tcp ...:22: i/o timeout`
- Usually SSH connectivity issue (host/firewall/security group/sshd)

## Data Attribution

- Source imagery: USDA NAIP
- Processed/labeled dataset: yinx111 (2025)
  https://github.com/yinx111/U-Net-Semantic-Segmentation-on-Multispectral-RGB-NIR-Imagery
