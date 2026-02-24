# U-Net++ MLOps Workflow

An end-to-end remote sensing segmentation project with:

- Local training with PyTorch + U-Net++
- DVC-based data/model artifact versioning
- MLflow experiment tracking and model registration
- FastAPI web inference app
- GitHub Actions CI/CD for remote deployment

## 1. Repository Structure

```text
.
├── app/                          # FastAPI web app
├── configs/train.yaml            # Training config
├── deploy/docker-compose.deploy.yml
├── scripts/
│   ├── train.py                  # Training
│   ├── evaluate.py               # Evaluation
│   ├── inference.py              # Inference + polygon export
│   ├── check_metrics.py          # Quality gate
│   └── register_model.py         # MLflow registration
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC lockfile (deployment-critical)
├── dataset_split.dvc             # DVC-tracked dataset pointer
├── docker-compose.yml            # Local app + cloudflared
└── .github/workflows/
    ├── ci.yml
    └── cd-app.yml
```

## 2. Prerequisites

- Python 3.9
- Linux + CUDA recommended for local GPU training

Install training/pipeline dependencies:

```bash
pip install -r requirements.txt
pip install "dvc[s3]==3.66.1"
```

If you only run the web app:

```bash
pip install -r requirements_app.txt
```

## 3. Local `.env` Setup

Create local env file:

```bash
cp .env.example .env
```

Recommended local values for training/inference:

- `CHECKPOINT_PATH=./outputs/model.pth`
- `IMG_PATH=./test_img/area_test1.tif`
- `OUT_DIR=./outputs/infer`
- GPU training: `FORCE_CPU=0`, `CUDA_VISIBLE_DEVICES=0` (or your preferred GPU list)

Important:
- Do not put deployment container paths (such as `/workspace/...`) into local training `.env`.
- Keep secrets out of Git (tokens, passwords, keys).

## 4. DVC Pipeline

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

# Typical model update stages
dvc repro train evaluate inference gate register

# Local status
dvc status

# Remote/cloud sync status
dvc status -c
```

## 5. MLflow Behavior

- Main config: `configs/train.yaml` -> `mlflow` section
- Default experiment name: `u-net-workflow`
- Each training run is appended as a new run under the same experiment
- If the experiment was soft-deleted, `train.py` restores it and keeps using the same name

## 6. Local Web Inference App

Local compose (`docker-compose.yml`) mounts local outputs and runs the app:

```bash
docker compose up -d --build
```

Endpoints:

- UI: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/healthz`

`/healthz` checks checkpoint existence/readability.

## 7. CI

`ci.yml` includes:

- Ruff + MyPy
- Pytest smoke tests
- DVC mini pipeline on `dataset_mini` (CPU)

## 8. CD (GitHub Actions)

Workflow: `.github/workflows/cd-app.yml`

### Trigger Rules

- `push` to `dev/main/master`: deploy only when `dvc.lock` changed
- `workflow_dispatch`: manual deploy to `staging`

### Deployment Flow

1. Build and push app image to GHCR
2. Runner executes `dvc pull` (default `outputs/model.pth`)
3. Runner uploads compose + model via SCP to target host
4. Remote host runs `docker compose up -d`
5. If health check fails, workflow rolls back to previous image

### Model Path Mapping

- Remote host: `<STAGING_DEPLOY_PATH>/deploy/models/<GITHUB_SHA>/model.pth`
- In container: `/workspace/models/<GITHUB_SHA>/model.pth`

Deployment compose file: `deploy/docker-compose.deploy.yml`

## 9. GitHub Environment (`staging`) Configuration

### Required Secrets

| Name | Required | Description | Example |
|---|---|---|---|
| `STAGING_SSH_HOST` | Yes | Public IP or DNS of deployment server. | `1.2.3.4` or `my-server.example.com` |
| `STAGING_SSH_USER` | Yes | SSH login user on target server. | `ubuntu` |
| `STAGING_SSH_KEY` | Yes | Private SSH key (PEM/OpenSSH) for `STAGING_SSH_USER`. | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `STAGING_DEPLOY_PATH` | Yes | Absolute directory on remote host where files are copied and compose runs. | `/home/ubuntu/webapp_deploy` |
| `STAGING_GHCR_USERNAME` | Yes | Username used to pull from GHCR on remote host. | `your-github-user` |
| `STAGING_GHCR_TOKEN` | Yes | Token/PAT with package read permission (`read:packages`). | `ghp_xxx...` |
| `STAGING_DVC_ACCESS_KEY_ID` | Yes | Access key for DVC S3-compatible remote (Dagshub storage). | `AKIA...` |
| `STAGING_DVC_SECRET_ACCESS_KEY` | Yes | Secret key for DVC S3-compatible remote. | `xxxx` |
| `CLOUDFLARED_TOKEN` | Yes | Cloudflare tunnel token used by `cloudflared` container. | `eyJ...` |

### Optional Variables

| Name | Required | Description | Default / Typical Value |
|---|---|---|---|
| `APP_HOST` | No | Host bind for FastAPI app in container env. | `0.0.0.0` |
| `APP_PORT` | No | Public/app port used by compose. | `8000` |
| `MAX_UPLOAD_MB` | No | Upload size limit for app endpoint. | `200` |
| `INFERENCE_TIMEOUT_SECONDS` | No | Inference subprocess timeout. | `3600` |
| `FORCE_CPU` | No | Force CPU inference in deployed app (`1/0`, `true/false`). | `1` |
| `CUDA_VISIBLE_DEVICES` | No | GPU visibility mask in deployed container. | empty |
| `MODEL_DVC_PATH` | No | DVC-tracked model file path pulled in CD. | `outputs/model.pth` |
| `MODEL_SHA256` | No | Optional integrity pin; CD fails if checksum mismatches. | empty |
| `DVC_AWS_REGION` | No | Region for DVC S3 client. | `us-east-1` |

Notes:
- `STAGING_DEPLOY_PATH` is configured as a secret in the current workflow.
- If `MODEL_SHA256` is empty, checksum validation is skipped.

## 10. Recommended Release Sequence

```bash
# 1) Train and update artifacts
dvc repro train evaluate inference gate register

# 2) Push model/artifacts to DVC remote
dvc push outputs/model.pth
# or dvc push

# 3) Commit code + dvc.lock
git add dvc.lock
git commit -m "update model"
git push
```

CD pulls the model version referenced by the commit's `dvc.lock`, not simply the newest object in remote storage.

## 11. Troubleshooting

1. `FileNotFoundError: /workspace/outputs/model.pth` during local runs
- Cause: local `.env` uses deployment container path
- Fix: set `CHECKPOINT_PATH=./outputs/model.pth`

2. `dvc status -c` fails with missing `dvc_s3`
- Fix: `pip install "dvc[s3]==3.66.1"`

3. CD SCP step fails with `dial tcp ...:22: i/o timeout`
- Usually SSH/network reachability issue (host, security group, firewall, or SSH service)

## 12. Data Attribution

- Source imagery: USDA NAIP
- Processed/labeled dataset: yinx111 (2025)
  https://github.com/yinx111/U-Net-Semantic-Segmentation-on-Multispectral-RGB-NIR-Imagery
