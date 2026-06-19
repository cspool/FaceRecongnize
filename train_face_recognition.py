import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


@dataclass
class ExperimentConfig:
    data_dir: str
    output_dir: str
    model: str
    public_weights: str
    seed: int
    image_size: int
    epochs: int
    batch_size: int
    val_fraction: float
    freeze_backbone_epochs: int
    backbone_lr: float
    head_lr: float
    weight_decay: float
    num_workers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a face recognizer on PIE official splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/PIE dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pie_resnet18"))
    parser.add_argument("--model", choices=["resnet18"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pie_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mat_paths = sorted(data_dir.glob("Pose*_64x64.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"No Pose*_64x64.mat files found in {data_dir}")

    images, labels, is_test, poses = [], [], [], []
    for path in mat_paths:
        mat = sio.loadmat(path)
        fea = mat["fea"].astype(np.uint8)
        gnd = mat["gnd"].reshape(-1).astype(np.int64) - 1
        test = mat["isTest"].reshape(-1).astype(bool)
        pose = path.stem.replace("_64x64", "")
        images.append(fea)
        labels.append(gnd)
        is_test.append(test)
        poses.append(np.full(gnd.shape, pose, dtype=object))

    return (
        np.concatenate(images, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(is_test, axis=0),
        np.concatenate(poses, axis=0),
    )


def make_splits(
    labels: np.ndarray,
    is_test: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trainval_idx = np.flatnonzero(~is_test)
    test_idx = np.flatnonzero(is_test)
    if not (0 < val_fraction < 1):
        return trainval_idx, np.array([], dtype=np.int64), test_idx

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    fit_rel, val_rel = next(splitter.split(np.zeros(len(trainval_idx)), labels[trainval_idx]))
    return trainval_idx[fit_rel], trainval_idx[val_rel], test_idx


class PIEDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        poses: np.ndarray,
        indices: np.ndarray,
        transform,
    ) -> None:
        self.images = images
        self.labels = labels
        self.poses = poses
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        image = Image.fromarray(self.images[idx].reshape(64, 64), mode="L")
        image = self.transform(image)
        return image, int(self.labels[idx]), str(self.poses[idx])


def make_transforms(image_size: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def build_model(name: str, num_classes: int) -> tuple[nn.Module, str]:
    if name != "resnet18":
        raise ValueError(f"Unsupported model: {name}")
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model, "torchvision.resnet18(IMAGENET1K_V1)"


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def make_optimizer(model: nn.Module, backbone_lr: float, head_lr: float, weight_decay: float):
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    scaler,
    device,
    amp_enabled: bool,
) -> dict:
    model.train()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    progress = tqdm(loader, desc="train", leave=False)
    for images, targets, _poses in progress:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.detach().item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size
        progress.set_postfix(loss=total_loss / total_examples, acc=total_correct / total_examples)

    return {
        "loss": total_loss / total_examples,
        "top1_acc": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device, amp_enabled: bool, num_classes: int) -> dict:
    model.eval()
    total_loss, total_examples = 0.0, 0
    y_true, y_pred, y_top5, pose_names = [], [], [], []

    for images, targets, poses in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        pred = logits.argmax(dim=1)
        k = min(5, num_classes)
        top5 = logits.topk(k=k, dim=1).indices

        y_true.extend(targets.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        y_top5.extend(top5.cpu().tolist())
        pose_names.extend(list(poses))

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    top5_correct = np.asarray([truth in pred_k for truth, pred_k in zip(y_true, y_top5)], dtype=bool)

    per_pose = {}
    for pose in sorted(set(pose_names)):
        mask = np.asarray([p == pose for p in pose_names])
        per_pose[pose] = {
            "count": int(mask.sum()),
            "top1_acc": float(accuracy_score(y_true_np[mask], y_pred_np[mask])),
            "top5_acc": float(top5_correct[mask].mean()),
        }

    return {
        "loss": total_loss / total_examples,
        "top1_acc": float(accuracy_score(y_true_np, y_pred_np)),
        "top5_acc": float(top5_correct.mean()),
        "per_pose": per_pose,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    images, labels, is_test, poses = load_pie_dataset(args.data_dir)
    num_classes = int(labels.max() + 1)
    train_idx, val_idx, test_idx = make_splits(labels, is_test, args.val_fraction, args.seed)
    train_tf, eval_tf = make_transforms(args.image_size)

    train_ds = PIEDataset(images, labels, poses, train_idx, train_tf)
    val_ds = PIEDataset(images, labels, poses, val_idx, eval_tf)
    test_ds = PIEDataset(images, labels, poses, test_idx, eval_tf)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model, public_weights = build_model(args.model, num_classes)
    model = model.to(device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)

    config = ExperimentConfig(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        model=args.model,
        public_weights=public_weights,
        seed=args.seed,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )
    split_summary = {
        "total": int(len(labels)),
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
        "num_classes": num_classes,
        "test_is_never_trained": True,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps({"config": asdict(config), "splits": split_summary}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"device": str(device), "splits": split_summary}, indent=2))

    best_val_acc = -1.0
    best_epoch = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        backbone_trainable = epoch > args.freeze_backbone_epochs
        set_backbone_trainable(model, backbone_trainable)
        print(f"\nEpoch {epoch}/{args.epochs} | backbone_trainable={backbone_trainable}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp_enabled)
        val_metrics = evaluate(model, val_loader, criterion, device, amp_enabled, num_classes)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1_acc": train_metrics["top1_acc"],
            "val_loss": val_metrics["loss"],
            "val_top1_acc": val_metrics["top1_acc"],
            "val_top5_acc": val_metrics["top5_acc"],
            "backbone_trainable": backbone_trainable,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "split_summary": split_summary,
                    "epoch": epoch,
                    "val_metrics": {k: v for k, v in val_metrics.items() if k not in {"y_true", "y_pred"}},
                },
                args.output_dir / "best_model.pt",
            )

    write_history(args.output_dir / "history.csv", history)
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = evaluate(model, val_loader, criterion, device, amp_enabled, num_classes)
    test_metrics = evaluate(model, test_loader, criterion, device, amp_enabled, num_classes)

    report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=list(range(num_classes)))
    np.savetxt(args.output_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    metrics = {
        "best_epoch": best_epoch,
        "elapsed_seconds": round(time.time() - t0, 3),
        "validation": {k: v for k, v in val_metrics.items() if k not in {"y_true", "y_pred"}},
        "test": {k: v for k, v in test_metrics.items() if k not in {"y_true", "y_pred"}},
        "classification_report": report,
        "split_summary": split_summary,
        "public_model": public_weights,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nFinal metrics")
    print(json.dumps({k: metrics[k] for k in ["best_epoch", "validation", "test", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
