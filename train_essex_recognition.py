import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


@dataclass
class Sample:
    path: str
    label_key: str
    subset: str


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
    test_fraction: float
    freeze_backbone_epochs: int
    backbone_lr: float
    head_lr: float
    weight_decay: float
    num_workers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a face recognizer on the Essex Face Recognition Data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/essex_fixed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/essex_resnet18"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to evaluate with --eval-only.")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used to initialize training. Incompatible classifier tensors are skipped.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Evaluate --checkpoint without training.")
    parser.add_argument("--model", choices=["resnet18"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.20)
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


def label_from_path(root: Path, path: Path) -> tuple[str, str]:
    rel = path.relative_to(root)
    subset = rel.parts[0]
    if subset == "faces94":
        if len(rel.parts) < 5:
            raise ValueError(f"Unexpected faces94 path: {path}")
        label_key = "/".join([subset, rel.parts[2], rel.parts[3]])
    else:
        if len(rel.parts) < 4:
            raise ValueError(f"Unexpected Essex path: {path}")
        label_key = "/".join([subset, rel.parts[2]])
    return subset, label_key


def discover_samples(data_dir: Path) -> tuple[list[Sample], list[dict]]:
    samples: list[Sample] = []
    skipped: list[dict] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        if ".xvpics" in path.parts:
            skipped.append({"path": str(path), "reason": "xvpics_thumbnail"})
            continue
        try:
            subset, label_key = label_from_path(data_dir, path)
            with Image.open(path) as image:
                image.convert("RGB").load()
        except Exception as exc:
            skipped.append({"path": str(path), "reason": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue
        samples.append(Sample(path=str(path), label_key=label_key, subset=subset))
    if not samples:
        raise RuntimeError(f"No valid Essex images found under {data_dir}")
    return samples, skipped


def make_label_map(samples: list[Sample]) -> dict[str, int]:
    return {label_key: idx for idx, label_key in enumerate(sorted({s.label_key for s in samples}))}


def make_splits(
    samples: list[Sample],
    label_to_id: dict[str, int],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        by_label[label_to_id[sample.label_key]].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label_id in sorted(by_label):
        indices = by_label[label_id]
        rng.shuffle(indices)
        n = len(indices)
        test_count = max(1, int(round(n * test_fraction)))
        val_count = max(1, int(round(n * val_fraction)))
        if n - test_count - val_count < 1:
            val_count = max(0, n - test_count - 1)
        test_idx.extend(indices[:test_count])
        val_idx.extend(indices[test_count : test_count + val_count])
        train_idx.extend(indices[test_count + val_count :])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


class EssexDataset(Dataset):
    def __init__(self, samples: list[Sample], indices: list[int], label_to_id: dict[str, int], transform) -> None:
        self.samples = samples
        self.indices = list(indices)
        self.label_to_id = label_to_id
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        sample = self.samples[self.indices[item]]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, self.label_to_id[sample.label_key], sample.subset


def make_transforms(image_size: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def build_model(name: str, num_classes: int, pretrained: bool = True) -> tuple[nn.Module, str]:
    if name != "resnet18":
        raise ValueError(f"Unsupported model: {name}")
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    public_weights = "torchvision.resnet18(IMAGENET1K_V1)" if pretrained else "none"
    return model, public_weights


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path, device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for name, tensor in state_dict.items():
        if name not in model_state:
            skipped.append({"name": name, "reason": "missing_in_model"})
            continue
        if tuple(model_state[name].shape) != tuple(tensor.shape):
            skipped.append(
                {
                    "name": name,
                    "reason": "shape_mismatch",
                    "checkpoint_shape": list(tensor.shape),
                    "model_shape": list(model_state[name].shape),
                }
            )
            continue
        compatible_state[name] = tensor
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "loaded_tensors": len(compatible_state),
        "skipped_tensors": skipped,
        "missing_after_load": list(missing),
        "unexpected_after_load": list(unexpected),
    }


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


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_enabled: bool) -> dict:
    model.train()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    for images, targets, _subsets in loader:
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

    return {
        "loss": total_loss / total_examples,
        "top1_acc": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled: bool, num_classes: int) -> dict:
    model.eval()
    total_loss, total_examples = 0.0, 0
    y_true, y_pred, y_top5, subsets = [], [], [], []
    for images, targets, batch_subsets in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        pred = logits.argmax(dim=1)
        top5 = logits.topk(k=min(5, num_classes), dim=1).indices
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        y_top5.extend(top5.cpu().tolist())
        subsets.extend(list(batch_subsets))

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    top5_correct = np.asarray([truth in pred_k for truth, pred_k in zip(y_true, y_top5)], dtype=bool)
    per_subset = {}
    for subset in sorted(set(subsets)):
        mask = np.asarray([s == subset for s in subsets])
        per_subset[subset] = {
            "count": int(mask.sum()),
            "top1_acc": float(accuracy_score(y_true_np[mask], y_pred_np[mask])),
            "top5_acc": float(top5_correct[mask].mean()),
        }

    return {
        "loss": total_loss / total_examples,
        "top1_acc": float(accuracy_score(y_true_np, y_pred_np)),
        "top5_acc": float(top5_correct.mean()),
        "per_subset": per_subset,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    keys = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def split_subset_summary(samples: list[Sample], indices: list[int]) -> dict[str, int]:
    return dict(sorted(Counter(samples[i].subset for i in indices).items()))


def write_eval_outputs(
    output_dir: Path,
    metrics_path: str,
    confusion_path: str,
    metrics: dict,
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    np.savetxt(output_dir / confusion_path, cm, fmt="%d", delimiter=",")
    metrics["classification_report"] = report
    (output_dir / metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    samples, skipped = discover_samples(args.data_dir)
    label_to_id = make_label_map(samples)
    train_idx, val_idx, test_idx = make_splits(
        samples=samples,
        label_to_id=label_to_id,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    num_classes = len(label_to_id)
    train_tf, eval_tf = make_transforms(args.image_size)
    train_ds = EssexDataset(samples, train_idx, label_to_id, train_tf)
    val_ds = EssexDataset(samples, val_idx, label_to_id, eval_tf)
    test_ds = EssexDataset(samples, test_idx, label_to_id, eval_tf)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    checkpoint_for_model = args.checkpoint if args.eval_only else args.init_checkpoint
    model, public_weights = build_model(args.model, num_classes, pretrained=checkpoint_for_model is None)
    model = model.to(device, memory_format=torch.channels_last)
    load_summary = None
    if checkpoint_for_model is not None:
        load_summary = load_model_checkpoint(model, checkpoint_for_model, device)
        print(json.dumps({"loaded_checkpoint": load_summary}, indent=2))
    criterion = nn.CrossEntropyLoss()

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
        test_fraction=args.test_fraction,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )
    split_summary = {
        "valid_images": len(samples),
        "skipped_images": len(skipped),
        "num_classes": num_classes,
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
        "train_by_subset": split_subset_summary(samples, train_idx),
        "val_by_subset": split_subset_summary(samples, val_idx),
        "test_by_subset": split_subset_summary(samples, test_idx),
        "test_is_never_trained": True,
    }
    write_csv(args.output_dir / "skipped_images.csv", skipped, fieldnames=["path", "reason"])
    (args.output_dir / "label_map.json").write_text(json.dumps(label_to_id, indent=2), encoding="utf-8")
    (args.output_dir / "config.json").write_text(
        json.dumps({"config": asdict(config), "splits": split_summary, "checkpoint_load": load_summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"device": str(device), "splits": split_summary}, indent=2))

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        val_metrics = evaluate(model, val_loader, criterion, device, amp_enabled, num_classes)
        test_metrics = evaluate(model, test_loader, criterion, device, amp_enabled, num_classes)
        eval_metrics = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_load": load_summary,
            "validation": {k: v for k, v in val_metrics.items() if k not in {"y_true", "y_pred"}},
            "test": {k: v for k, v in test_metrics.items() if k not in {"y_true", "y_pred"}},
            "split_summary": split_summary,
            "public_model": public_weights,
        }
        write_eval_outputs(
            args.output_dir,
            "eval_metrics.json",
            "eval_confusion_matrix.csv",
            eval_metrics,
            test_metrics["y_true"],
            test_metrics["y_pred"],
            num_classes,
        )
        print("\nEvaluation metrics")
        print(json.dumps({k: eval_metrics[k] for k in ["validation", "test"]}, indent=2))
        return

    optimizer = make_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)

    best_val_acc = -1.0
    best_epoch = 0
    history = []
    t0 = time.time()
    initial_val_metrics = None
    if args.init_checkpoint is not None:
        initial_val_metrics = evaluate(model, val_loader, criterion, device, amp_enabled, num_classes)
        best_val_acc = initial_val_metrics["top1_acc"]
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "split_summary": split_summary,
                "epoch": 0,
                "val_metrics": {k: v for k, v in initial_val_metrics.items() if k not in {"y_true", "y_pred"}},
                "label_to_id": label_to_id,
                "checkpoint_load": load_summary,
            },
            args.output_dir / "best_model.pt",
        )
        history.append(
            {
                "epoch": 0,
                "train_loss": "",
                "train_top1_acc": "",
                "val_loss": initial_val_metrics["loss"],
                "val_top1_acc": initial_val_metrics["top1_acc"],
                "val_top5_acc": initial_val_metrics["top5_acc"],
                "backbone_trainable": True,
            }
        )
    for epoch in range(1, args.epochs + 1):
        backbone_trainable = epoch > args.freeze_backbone_epochs
        set_backbone_trainable(model, backbone_trainable)
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
                    "label_to_id": label_to_id,
                },
                args.output_dir / "best_model.pt",
            )

    write_csv(args.output_dir / "history.csv", history)
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
        "initial_validation": None
        if initial_val_metrics is None
        else {k: v for k, v in initial_val_metrics.items() if k not in {"y_true", "y_pred"}},
        "validation": {k: v for k, v in val_metrics.items() if k not in {"y_true", "y_pred"}},
        "test": {k: v for k, v in test_metrics.items() if k not in {"y_true", "y_pred"}},
        "classification_report": report,
        "split_summary": split_summary,
        "public_model": public_weights,
        "checkpoint_load": load_summary,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nFinal metrics")
    print(json.dumps({k: metrics[k] for k in ["best_epoch", "validation", "test", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
