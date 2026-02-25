import argparse
import json
import os
import random
import subprocess
from pathlib import Path

from typing import Dict, Any

import albumentations as A
import cv2
import mlflow
import numpy as np
import segmentation_models_pytorch as smp
import tifffile as tiff
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from env_loader import load_dotenv

load_dotenv()

# Your paths
DATA_ROOT = os.getenv("DATA_ROOT", "./dataset_split")
OUTPUT_DIR = "./outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pth")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "log.txt")
DEFAULT_CONFIG_PATH = "configs/train.yaml"

# MLflow
MLFLOW_ENABLED = False
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "u-net-workflow")
MLFLOW_RUN_NAME = None
MLFLOW_TAGS: Dict[str, Any] = {}
MLFLOW_ARTIFACT_SUBDIR = "artifacts"

# Training hyperparameters
NUM_CLASSES = 6
IN_CHANNELS = 4
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42
# Respect env override for CPU-only runs (e.g., CI)
FORCE_CPU = os.getenv("FORCE_CPU", "0").lower() in {"1", "true", "yes"}
DEVICE = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
AMP = True  # Mixed precision
SAVE_BEST = True
IGNORE_INDEX = -1

# Choose encoder
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = None  # do not load ImageNet
DECODER_USE_BATCHNORM = True


# Random seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not FORCE_CPU:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# Config helpers
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_config(cfg: dict):
    global DATA_ROOT, OUTPUT_DIR, MODEL_PATH, METRICS_PATH, LOG_FILE
    global NUM_CLASSES, IN_CHANNELS, IMG_SIZE, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, NUM_WORKERS, SEED, AMP, SAVE_BEST, IGNORE_INDEX
    global ENCODER_NAME, ENCODER_WEIGHTS, DECODER_USE_BATCHNORM
    global MLFLOW_ENABLED, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MLFLOW_RUN_NAME, MLFLOW_TAGS, MLFLOW_ARTIFACT_SUBDIR

    DATA_ROOT = cfg.get("data", {}).get("root", DATA_ROOT)

    paths_cfg = cfg.get("paths", {})
    OUTPUT_DIR = paths_cfg.get("output_dir", OUTPUT_DIR)
    MODEL_PATH = paths_cfg.get("model_path", os.path.join(OUTPUT_DIR, "model.pth"))
    METRICS_PATH = paths_cfg.get("metrics_path", os.path.join(OUTPUT_DIR, "metrics.json"))
    LOG_FILE = paths_cfg.get("log_file", os.path.join(OUTPUT_DIR, "log.txt"))

    model_cfg = cfg.get("model", {})
    ENCODER_NAME = model_cfg.get("encoder_name", ENCODER_NAME)
    ENCODER_WEIGHTS = model_cfg.get("encoder_weights", ENCODER_WEIGHTS)
    DECODER_USE_BATCHNORM = model_cfg.get("decoder_use_batchnorm", DECODER_USE_BATCHNORM)
    IN_CHANNELS = model_cfg.get("in_channels", IN_CHANNELS)
    NUM_CLASSES = model_cfg.get("num_classes", NUM_CLASSES)

    train_cfg = cfg.get("training", {})
    IMG_SIZE = train_cfg.get("img_size", IMG_SIZE)
    BATCH_SIZE = train_cfg.get("batch_size", BATCH_SIZE)
    EPOCHS = train_cfg.get("epochs", EPOCHS)
    LR = train_cfg.get("lr", LR)
    WEIGHT_DECAY = train_cfg.get("weight_decay", WEIGHT_DECAY)
    NUM_WORKERS = train_cfg.get("num_workers", NUM_WORKERS)
    SEED = train_cfg.get("seed", SEED)
    AMP = train_cfg.get("amp", AMP)
    SAVE_BEST = train_cfg.get("save_best", SAVE_BEST)
    IGNORE_INDEX = train_cfg.get("ignore_index", IGNORE_INDEX)

    mlflow_cfg = cfg.get("mlflow", {})
    MLFLOW_ENABLED = mlflow_cfg.get("enabled", MLFLOW_ENABLED)
    MLFLOW_TRACKING_URI = mlflow_cfg.get("tracking_uri", MLFLOW_TRACKING_URI)
    MLFLOW_EXPERIMENT = mlflow_cfg.get("experiment", MLFLOW_EXPERIMENT)
    MLFLOW_RUN_NAME = mlflow_cfg.get("run_name", MLFLOW_RUN_NAME)
    MLFLOW_TAGS = mlflow_cfg.get("tags", MLFLOW_TAGS) or {}
    MLFLOW_ARTIFACT_SUBDIR = mlflow_cfg.get("artifact_subdir", MLFLOW_ARTIFACT_SUBDIR)


def apply_env_overrides():
    global DATA_ROOT, BATCH_SIZE, EPOCHS, NUM_WORKERS, AMP, MLFLOW_ENABLED
    env_data_root = os.getenv("DATA_ROOT")
    if env_data_root:
        DATA_ROOT = env_data_root
    env_batch = os.getenv("BATCH_SIZE")
    if env_batch:
        BATCH_SIZE = int(env_batch)
    env_epochs = os.getenv("EPOCHS")
    if env_epochs:
        EPOCHS = int(env_epochs)
    env_workers = os.getenv("NUM_WORKERS")
    if env_workers:
        NUM_WORKERS = int(env_workers)
    env_amp = os.getenv("AMP")
    if env_amp is not None:
        AMP = env_amp.lower() in {"1", "true", "yes"}
    env_mlflow = os.getenv("MLFLOW_ENABLED")
    if env_mlflow is not None:
        MLFLOW_ENABLED = env_mlflow.lower() in {"1", "true", "yes"}


def ensure_log_header():
    header = "epoch\ttrain_loss\tval_loss\tval_mIoU\tval_OA\n"
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(header)
        return
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        first_line = f.readline()
        rest = f.read()
    if not first_line.startswith("epoch"):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(first_line)
            f.write(rest)


# MLflow helpers
def mlflow_safe(callable_fn, *args, **kwargs):
    try:
        return callable_fn(*args, **kwargs)
    except Exception as e:
        print(f"[MLflow] Warning: {e}")
        return None


def get_git_commit_hash():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def mlflow_start():
    if not MLFLOW_ENABLED:
        return None, None
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    except Exception as e:
        # If the target experiment was soft-deleted, restore it and reuse the same name.
        msg = str(e).lower()
        if "deleted experiment" in msg:
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
            if exp is not None and getattr(exp, "lifecycle_stage", "") == "deleted":
                client.restore_experiment(exp.experiment_id)
                print(f"[MLflow] Restored deleted experiment: {MLFLOW_EXPERIMENT}")
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
        else:
            raise
    commit = get_git_commit_hash()
    tags = dict(MLFLOW_TAGS)
    if commit:
        tags.setdefault("git_commit", commit)
    run = mlflow.start_run(run_name=MLFLOW_RUN_NAME, tags=tags)
    return run, commit


def mlflow_log_params_from_cfg():
    if not MLFLOW_ENABLED:
        return
    params = {
        "data.root": DATA_ROOT,
        "model.encoder": ENCODER_NAME,
        "model.encoder_weights": ENCODER_WEIGHTS,
        "model.decoder_bn": DECODER_USE_BATCHNORM,
        "model.in_channels": IN_CHANNELS,
        "model.num_classes": NUM_CLASSES,
        "train.img_size": IMG_SIZE,
        "train.batch_size": BATCH_SIZE,
        "train.epochs": EPOCHS,
        "train.lr": LR,
        "train.weight_decay": WEIGHT_DECAY,
        "train.num_workers": NUM_WORKERS,
        "train.seed": SEED,
        "train.amp": AMP,
        "train.save_best": SAVE_BEST,
        "train.ignore_index": IGNORE_INDEX,
    }
    mlflow_safe(mlflow.log_params, params)


# Dataset utilities
IMG_PREFIX = "tile_"
MASK_PREFIX = "mask_"
EXTS_IMG = (".tif", ".tiff", ".png")
EXTS_MASK = (".tif", ".tiff", ".png")


def list_pairs(split_dir: str):
    img_dir = Path(split_dir) / "img"
    mask_dir = Path(split_dir) / "mask"
    # Build mask lookup
    mask_map = {}
    for p in mask_dir.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS_MASK and p.name.startswith(MASK_PREFIX):
            _id = p.stem[len(MASK_PREFIX) :]
            mask_map[_id] = p
    # Match images
    pairs = []
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS_IMG and p.name.startswith(IMG_PREFIX):
            _id = p.stem[len(IMG_PREFIX) :]
            if _id in mask_map:
                pairs.append((str(p), str(mask_map[_id])))
    return sorted(pairs)


def load_image_4ch(path: str) -> np.ndarray:
    arr = tiff.imread(path)  # Could be (H,W,C) or (C,H,W)
    if arr.ndim == 2:
        raise RuntimeError(f"Single-channel image: {path}")
    if arr.shape[0] in (3, 4, 5) and arr.ndim == 3 and arr.shape[0] < arr.shape[1]:
        # (C,H,W) -> (H,W,C)
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] < 4:
        raise RuntimeError(f"Insufficient channels (<4): {path}, shape={arr.shape}")
    if arr.shape[2] > 4:
        arr = arr[:, :, :4]

    # Normalize
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)

        if arr.max() > 1.0:
            arr = np.clip(arr / (arr.max() + 1e-6), 0.0, 1.0)
    return arr


def load_mask_class(path: str) -> np.ndarray:

    m = tiff.imread(path)
    if m.ndim == 3:
        # Allow single-channel saved as (H,W,1)
        if m.shape[2] == 1:
            m = m[:, :, 0]
        else:
            raise RuntimeError(f"Mask should not be multi-channel: {path}, shape={m.shape}")
    # If float, convert to int
    if np.issubdtype(m.dtype, np.floating):
        m = np.rint(m).astype(np.int64)
    else:
        m = m.astype(np.int64)
    return m


# Albumentations augmentations
def get_train_augs(img_size=256):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Use Affine instead of ShiftScaleRotate; use cval/cval_mask instead of value/mask_value
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-15, 15),
                interpolation=cv2.INTER_LINEAR,  # image interpolation
                mask_interpolation=cv2.INTER_NEAREST,  # mask must be nearest
                fit_output=False,
                p=0.5,
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )


def get_val_augs(img_size=256):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            ToTensorV2(transpose_mask=True),
        ]
    )


# Dataset
class RSDataset(Dataset):
    def __init__(self, pairs, transforms=None):
        self.items = pairs
        self.tf = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        image = load_image_4ch(img_path)
        mask = load_mask_class(mask_path)

        # albumentations expects: image(H,W,C) mask(H,W)
        if self.tf is not None:
            out = self.tf(image=image, mask=mask)
            image = out["image"]  # tensor (C,H,W)
            mask = out["mask"]  # tensor (H,W)
        # Ensure dtype
        mask = mask.long()
        return image, mask, img_path


# Model
def build_model():
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        decoder_use_batchnorm=DECODER_USE_BATCHNORM,
    )
    return model


def compute_iou(pred, target, num_classes=6, ignore_index=None):
    if ignore_index is not None:
        valid = target != ignore_index
        pred = pred[valid]
        target = target[valid]
    if target.numel() == 0:
        return 0.0

    ious = []
    for cls in range(num_classes):
        pred_c = pred == cls
        targ_c = target == cls
        inter = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()
        if union == 0:
            continue
        ious.append(inter / (union + 1e-7))
    if not ious:
        return 0.0
    return float(np.mean(ious))


def mlflow_log_epoch(epoch, train_loss, val_loss, val_miou, val_oa):
    if not MLFLOW_ENABLED:
        return
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_mIoU": val_miou,
        "val_OA": val_oa,
    }
    mlflow_safe(mlflow.log_metrics, metrics, step=epoch)


def mlflow_log_final(best_epoch, best_miou, best_val_loss, test_loss, test_miou, test_oa):
    if not MLFLOW_ENABLED:
        return
    metrics = {
        "best_epoch": best_epoch,
        "best_val_mIoU": best_miou,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_mIoU": test_miou,
        "test_OA": test_oa,
    }
    mlflow_safe(mlflow.log_metrics, metrics)


def mlflow_log_artifacts(artifact_paths):
    if not MLFLOW_ENABLED:
        return
    for p in artifact_paths:
        if p and os.path.exists(p):
            mlflow_safe(mlflow.log_artifact, p, artifact_path=MLFLOW_ARTIFACT_SUBDIR)


def mlflow_log_model(model):
    if not MLFLOW_ENABLED:
        return None
    import mlflow.pytorch as mlflow_pytorch

    info = mlflow_pytorch.log_model(model, artifact_path="model")
    return getattr(info, "model_uri", None)


# Train
def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for images, masks, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=AMP):
            logits = model(images)
            loss = criterion(logits, masks)

        if AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(1, n_batches)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    n_batches = 0
    miou_sum = 0.0
    n_samples = 0
    correct = 0
    total = 0
    for images, masks, _ in tqdm(loader, desc="Val", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=AMP):
            logits = model(images)
            loss = criterion(logits, masks)

        running_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=1)  # (N,H,W)
        preds_cpu = preds.cpu()
        masks_cpu = masks.cpu()

        miou_sum += compute_iou(
            preds_cpu, masks_cpu, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX
        )
        if IGNORE_INDEX is not None:
            valid = masks_cpu != IGNORE_INDEX
            correct += (preds_cpu[valid] == masks_cpu[valid]).sum().item()
            total += valid.sum().item()
        else:
            correct += (preds_cpu == masks_cpu).sum().item()
            total += masks_cpu.numel()
        n_samples += 1
    overall_acc = correct / max(1, total)
    return running_loss / max(1, n_batches), miou_sum / max(1, n_samples), overall_acc


# Main
def main(cfg_path: str = DEFAULT_CONFIG_PATH):
    cfg = load_config(cfg_path)
    apply_config(cfg)
    apply_env_overrides()
    if FORCE_CPU:
        # Ensure CPU-only runs don't attempt AMP
        globals()["AMP"] = False
    set_seed(SEED)

    active_run, git_commit = mlflow_start()
    mlflow_log_params_from_cfg()
    if MLFLOW_ENABLED and git_commit:
        mlflow_safe(mlflow.log_param, "git_commit", git_commit)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_log_header()

    train_pairs = list_pairs(os.path.join(DATA_ROOT, "train"))
    val_pairs = list_pairs(os.path.join(DATA_ROOT, "val"))
    test_pairs = list_pairs(os.path.join(DATA_ROOT, "test"))

    if MLFLOW_ENABLED:
        mlflow_safe(
            mlflow.log_params,
            {
                "data.train_size": len(train_pairs),
                "data.val_size": len(val_pairs),
                "data.test_size": len(test_pairs),
            },
        )

    print(f"[Info] train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    train_ds = RSDataset(train_pairs, transforms=get_train_augs(IMG_SIZE))
    val_ds = RSDataset(val_pairs, transforms=get_val_augs(IMG_SIZE))
    test_ds = RSDataset(test_pairs, transforms=get_val_augs(IMG_SIZE))

    pin_mem = DEVICE == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
    )

    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best_miou = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_path = MODEL_PATH

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        val_loss, val_miou, val_oa = eval_one_epoch(model, val_loader, criterion)
        scheduler.step()

        print(
            f"[Epoch {epoch}] "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_mIoU={val_miou:.4f} | "
            f"val_OA={val_oa:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        mlflow_log_epoch(epoch, tr_loss, val_loss, val_miou, val_oa)

        # Log and save
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{tr_loss:.6f}\t{val_loss:.6f}\t{val_miou:.6f}\t{val_oa:.6f}\n")

        if SAVE_BEST and val_miou > best_miou:
            best_miou = val_miou
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_mIoU": best_miou,
                    "val_loss": best_val_loss,
                    "config": {
                        "in_channels": IN_CHANNELS,
                        "num_classes": NUM_CLASSES,
                        "encoder": ENCODER_NAME,
                    },
                },
                best_path,
            )
            print(f"[Save] New best mIoU={best_miou:.4f} -> {best_path}")

    if not os.path.exists(best_path):
        torch.save(
            {
                "epoch": EPOCHS,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_mIoU": best_miou,
                "val_loss": best_val_loss,
                "config": {
                    "in_channels": IN_CHANNELS,
                    "num_classes": NUM_CLASSES,
                    "encoder": ENCODER_NAME,
                },
            },
            best_path,
        )
        print(f"[Save] Saved final model -> {best_path}")

    # Test set evaluation
    print("\n[Eval] Testing best model on test set ...")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
    test_loss, test_miou, test_oa = eval_one_epoch(model, test_loader, criterion)
    print(f"[Test] loss={test_loss:.4f} | mIoU={test_miou:.4f} | OA={test_oa:.4f}")

    # Log best model to MLflow if enabled
    mlflow_model_uri = mlflow_log_model(model)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_mIoU": best_miou,
        "best_val_loss": best_val_loss,
        "test_mIoU": test_miou,
        "test_OA": test_oa,
        "test_loss": test_loss,
        "config": {
            "in_channels": IN_CHANNELS,
            "num_classes": NUM_CLASSES,
            "encoder": ENCODER_NAME,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "amp": AMP,
        },
    }
    if active_run is not None:
        metrics["mlflow_run_id"] = active_run.info.run_id
    if mlflow_model_uri is not None:
        metrics["mlflow_model_uri"] = mlflow_model_uri
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved to {METRICS_PATH}")

    mlflow_log_final(best_epoch, best_miou, best_val_loss, test_loss, test_miou, test_oa)
    mlflow_log_artifacts([METRICS_PATH, LOG_FILE, best_path, cfg_path])

    if active_run is not None:
        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unet++ for semantic segmentation")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
