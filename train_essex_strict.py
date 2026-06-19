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
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from train_essex_recognition import Sample, discover_samples


@dataclass
class StrictConfig:
    data_dir: str
    output_dir: str
    model: str
    public_weights: str
    protocol: str
    hard_subsets: list[str]
    seed: int
    image_size: int
    epochs: int
    batch_size: int
    hard_val_fraction: float
    hard_test_fraction: float
    train_fractions: dict[str, float]
    hard_train_fractions: dict[str, float]
    no_hard_train: bool
    random_weights: bool
    eval_random_only: bool
    backbone_lr: float
    head_lr: float
    weight_decay: float
    num_workers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict FRD/Essex training with subject-disjoint hard-subset evaluation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/essex"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/essex_strict_hard_resnet18"))
    parser.add_argument("--model", choices=["resnet18"], default="resnet18")
    parser.add_argument("--hard-subsets", nargs="+", default=["faces96", "grimace"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hard-val-fraction", type=float, default=0.20)
    parser.add_argument("--hard-test-fraction", type=float, default=0.20)
    parser.add_argument(
        "--train-fractions",
        nargs="*",
        default=[],
        metavar="SUBSET=FRACTION",
        help="Per-subset identity fraction assigned to training; unused non-hard identities are excluded.",
    )
    parser.add_argument(
        "--hard-train-fractions",
        nargs="*",
        default=[],
        metavar="SUBSET=FRACTION",
        help="Per-hard-subset identity fraction assigned to training before splitting the remainder into val/test.",
    )
    parser.add_argument(
        "--no-hard-train",
        action="store_true",
        help="Use hard subsets only for validation/test; hard identities not assigned to val/test are excluded.",
    )
    parser.add_argument("--random-weights", action="store_true", help="Initialize the model without ImageNet weights.")
    parser.add_argument(
        "--eval-random-only",
        action="store_true",
        help="Evaluate a randomly initialized model without training.",
    )
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.train_fractions = parse_fraction_map(args.train_fractions)
    args.hard_train_fractions = parse_fraction_map(args.hard_train_fractions)
    if args.eval_random_only:
        args.random_weights = True
    return args


def parse_fraction_map(items: list[str]) -> dict[str, float]:
    fractions = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected SUBSET=FRACTION, got: {item}")
        subset, value_text = item.split("=", 1)
        subset = subset.strip()
        if not subset:
            raise ValueError(f"Empty subset in fraction spec: {item}")
        value = float(value_text)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Fraction for {subset} must be in [0, 1], got {value}")
        fractions[subset] = value
    return fractions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_identity_splits(
    samples: list[Sample],
    hard_subsets: list[str],
    hard_val_fraction: float,
    hard_test_fraction: float,
    train_fractions: dict[str, float],
    hard_train_fractions: dict[str, float],
    no_hard_train: bool,
    seed: int,
) -> dict:
    hard_set = set(hard_subsets)
    ids_by_subset: dict[str, list[str]] = defaultdict(list)
    for label_key in sorted({s.label_key for s in samples}):
        subset = label_key.split("/", 1)[0]
        ids_by_subset[subset].append(label_key)

    rng = random.Random(seed)
    train_ids, val_ids, test_ids, excluded_ids = [], [], [], []
    per_subset = {}

    for subset in sorted(ids_by_subset):
        ids = list(ids_by_subset[subset])
        rng.shuffle(ids)
        if subset in hard_set:
            n = len(ids)
            if subset in hard_train_fractions:
                train_count = int(round(n * hard_train_fractions[subset]))
                if n - train_count < 2:
                    train_count = max(0, n - 2)
                subset_train = ids[:train_count]
                remaining = ids[train_count:]
                val_test_fraction = hard_val_fraction + hard_test_fraction
                if val_test_fraction <= 0:
                    raise ValueError("--hard-val-fraction + --hard-test-fraction must be > 0")
                test_count = int(round(len(remaining) * hard_test_fraction / val_test_fraction))
                test_count = min(max(test_count, 1), len(remaining) - 1)
                subset_test = remaining[:test_count]
                subset_val = remaining[test_count:]
                subset_excluded = []
            else:
                test_count = max(1, int(round(n * hard_test_fraction)))
                val_count = max(1, int(round(n * hard_val_fraction)))
                min_remaining = 0 if no_hard_train else 1
                if n - test_count - val_count < min_remaining:
                    val_count = max(0, n - test_count - 1)
                subset_test = ids[:test_count]
                subset_val = ids[test_count : test_count + val_count]
                remaining = ids[test_count + val_count :]
                subset_train = [] if no_hard_train else remaining
                subset_excluded = remaining if no_hard_train else []
            test_ids.extend(subset_test)
            val_ids.extend(subset_val)
            train_ids.extend(subset_train)
            excluded_ids.extend(subset_excluded)
        else:
            train_fraction = train_fractions.get(subset, 1.0)
            train_count = int(round(len(ids) * train_fraction))
            if train_fraction > 0.0:
                train_count = max(1, train_count)
            train_count = min(train_count, len(ids))
            subset_train = ids[:train_count]
            subset_val = []
            subset_test = []
            subset_excluded = ids[train_count:]
            train_ids.extend(subset_train)
            excluded_ids.extend(subset_excluded)

        per_subset[subset] = {
            "train_identities": len(subset_train),
            "val_identities": len(subset_val),
            "test_identities": len(subset_test),
            "excluded_identities": len(subset_excluded),
        }

    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)
    test_ids = sorted(test_ids)
    overlap = {
        "train_val": sorted(set(train_ids) & set(val_ids)),
        "train_test": sorted(set(train_ids) & set(test_ids)),
        "val_test": sorted(set(val_ids) & set(test_ids)),
    }
    return {
        "protocol": "hard_subject_disjoint_embedding",
        "hard_subsets": sorted(hard_set),
        "train_fractions": train_fractions,
        "hard_train_fractions": hard_train_fractions,
        "train_identities": train_ids,
        "val_identities": val_ids,
        "test_identities": test_ids,
        "excluded_identities": sorted(excluded_ids),
        "per_subset": per_subset,
        "overlap": overlap,
    }


def indices_for_identities(samples: list[Sample], identities: list[str]) -> list[int]:
    identity_set = set(identities)
    return [idx for idx, sample in enumerate(samples) if sample.label_key in identity_set]


class TrainIdentityDataset(Dataset):
    def __init__(self, samples: list[Sample], indices: list[int], label_to_class: dict[str, int], transform) -> None:
        self.samples = samples
        self.indices = list(indices)
        self.label_to_class = label_to_class
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        sample = self.samples[self.indices[item]]
        with Image.open(sample.path) as image:
            image = self.transform(image.convert("RGB"))
        return image, self.label_to_class[sample.label_key]


class EvalIdentityDataset(Dataset):
    def __init__(self, samples: list[Sample], indices: list[int], transform) -> None:
        self.samples = samples
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        sample = self.samples[self.indices[item]]
        with Image.open(sample.path) as image:
            image = self.transform(image.convert("RGB"))
        return image, sample.label_key, sample.subset, sample.path


def make_transforms(image_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.10),
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


def build_model(name: str, num_classes: int, random_weights: bool = False) -> tuple[nn.Module, str]:
    if name != "resnet18":
        raise ValueError(f"Unsupported model: {name}")
    weights = None if random_weights else models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    public_weights = "random_initialization" if random_weights else "torchvision.resnet18(IMAGENET1K_V1)"
    return model, public_weights


def extract_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return F.normalize(x, dim=1)


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
    for images, targets in tqdm(loader, desc="train", leave=False):
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

    return {"loss": total_loss / total_examples, "top1_acc": total_correct / total_examples}


@torch.no_grad()
def collect_embeddings(model, loader, device, amp_enabled: bool) -> dict:
    model.eval()
    embeddings, labels, subsets, paths = [], [], [], []
    for images, batch_labels, batch_subsets, batch_paths in tqdm(loader, desc="embed", leave=False):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            feats = extract_features(model, images)
        embeddings.append(feats.float().cpu().numpy())
        labels.extend(list(batch_labels))
        subsets.extend(list(batch_subsets))
        paths.extend(list(batch_paths))
    return {
        "embeddings": np.concatenate(embeddings, axis=0),
        "labels": np.asarray(labels, dtype=object),
        "subsets": np.asarray(subsets, dtype=object),
        "paths": np.asarray(paths, dtype=object),
    }


def pair_scores(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sims = embeddings @ embeddings.T
    upper = np.triu_indices(len(labels), k=1)
    scores = sims[upper]
    same = labels[upper[0]] == labels[upper[1]]
    return scores.astype(np.float64), same.astype(bool)


def threshold_metrics(scores: np.ndarray, same: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    tp = int(np.logical_and(pred, same).sum())
    fp = int(np.logical_and(pred, ~same).sum())
    tn = int(np.logical_and(~pred, ~same).sum())
    fn = int(np.logical_and(~pred, same).sum())
    far = fp / (fp + tn) if fp + tn else 0.0
    tar = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(same)),
        "balanced_accuracy": float((tar + (1.0 - far)) / 2.0),
        "precision": float(precision),
        "recall_tar": float(tar),
        "f1": float(f1),
        "far": float(far),
        "true_accepts": tp,
        "false_accepts": fp,
        "true_rejects": tn,
        "false_rejects": fn,
    }


def best_balanced_threshold(scores: np.ndarray, same: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(same, scores)
    balanced = (tpr + (1.0 - fpr)) / 2.0
    finite = np.isfinite(thresholds)
    if not finite.any():
        return float(np.max(scores))
    masked_balanced = np.where(finite, balanced, -np.inf)
    return float(thresholds[int(np.argmax(masked_balanced))])


def threshold_for_far(scores: np.ndarray, same: np.ndarray, target_far: float) -> float:
    negative_scores = np.sort(scores[~same])
    if len(negative_scores) == 0:
        return float(np.max(scores) + 1e-6)
    index = int(np.ceil((1.0 - target_far) * len(negative_scores))) - 1
    index = min(max(index, 0), len(negative_scores) - 1)
    return float(negative_scores[index])


def roc_metrics(scores: np.ndarray, same: np.ndarray) -> dict:
    auc = roc_auc_score(same, scores)
    fpr, tpr, thresholds = roc_curve(same, scores)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return {
        "auc": float(auc),
        "eer": float((fpr[idx] + fnr[idx]) / 2.0),
        "eer_threshold": float(thresholds[idx]),
        "positive_pairs": int(same.sum()),
        "negative_pairs": int((~same).sum()),
    }


def verification_report(eval_data: dict, val_thresholds: dict | None = None) -> tuple[dict, dict]:
    scores, same = pair_scores(eval_data["embeddings"], eval_data["labels"])
    report = roc_metrics(scores, same)
    if val_thresholds is None:
        thresholds = {
            "best_balanced": best_balanced_threshold(scores, same),
            "far_1pct": threshold_for_far(scores, same, 0.01),
            "far_0_1pct": threshold_for_far(scores, same, 0.001),
        }
    else:
        thresholds = val_thresholds

    report["thresholds"] = {name: threshold_metrics(scores, same, value) for name, value in thresholds.items()}
    return report, thresholds


def identification_report(eval_data: dict) -> dict:
    embeddings = eval_data["embeddings"]
    labels = eval_data["labels"]
    subsets = eval_data["subsets"]
    paths = eval_data["paths"]

    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[str(label)].append(idx)
    gallery_indices, probe_indices = [], []
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda idx: str(paths[idx]))
        gallery_indices.append(ordered[0])
        probe_indices.extend(ordered[1:])

    gallery_embeddings = embeddings[gallery_indices]
    gallery_labels = labels[gallery_indices]
    probe_embeddings = embeddings[probe_indices]
    probe_labels = labels[probe_indices]
    probe_subsets = subsets[probe_indices]

    sims = probe_embeddings @ gallery_embeddings.T
    rank_order = np.argsort(-sims, axis=1)
    top1 = gallery_labels[rank_order[:, 0]] == probe_labels
    top5 = np.asarray(
        [probe_labels[i] in set(gallery_labels[rank_order[i, : min(5, len(gallery_labels))]]) for i in range(len(probe_labels))]
    )

    per_subset = {}
    for subset in sorted(set(probe_subsets)):
        mask = probe_subsets == subset
        per_subset[str(subset)] = {
            "probe_images": int(mask.sum()),
            "top1_acc": float(top1[mask].mean()),
            "top5_acc": float(top5[mask].mean()),
        }

    return {
        "gallery_identities": int(len(gallery_indices)),
        "probe_images": int(len(probe_indices)),
        "top1_acc": float(top1.mean()),
        "top5_acc": float(top5.mean()),
        "per_subset": per_subset,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_summary(samples: list[Sample], split_indices: dict[str, list[int]], identity_splits: dict) -> dict:
    summary = {
        "identity_counts": {
            "train": len(identity_splits["train_identities"]),
            "val": len(identity_splits["val_identities"]),
            "test": len(identity_splits["test_identities"]),
            "excluded": len(identity_splits["excluded_identities"]),
        },
        "image_counts": {name: len(indices) for name, indices in split_indices.items()},
        "image_counts_by_subset": {},
        "identity_counts_by_subset": identity_splits["per_subset"],
        "overlap": identity_splits["overlap"],
        "test_identities_never_trained": len(identity_splits["overlap"]["train_test"]) == 0,
        "val_identities_never_trained": len(identity_splits["overlap"]["train_val"]) == 0,
    }
    for name, indices in split_indices.items():
        summary["image_counts_by_subset"][name] = dict(sorted(Counter(samples[i].subset for i in indices).items()))
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    samples, skipped = discover_samples(args.data_dir)
    identity_splits = make_identity_splits(
        samples,
        hard_subsets=args.hard_subsets,
        hard_val_fraction=args.hard_val_fraction,
        hard_test_fraction=args.hard_test_fraction,
        train_fractions=args.train_fractions,
        hard_train_fractions=args.hard_train_fractions,
        no_hard_train=args.no_hard_train,
        seed=args.seed,
    )
    split_indices = {
        "train": indices_for_identities(samples, identity_splits["train_identities"]),
        "val": indices_for_identities(samples, identity_splits["val_identities"]),
        "test": indices_for_identities(samples, identity_splits["test_identities"]),
        "excluded": indices_for_identities(samples, identity_splits["excluded_identities"]),
    }
    summary = split_summary(samples, split_indices, identity_splits)
    if not summary["test_identities_never_trained"] or not summary["val_identities_never_trained"]:
        raise RuntimeError(f"Identity leakage detected: {summary['overlap']}")

    train_label_to_class = {label: idx for idx, label in enumerate(identity_splits["train_identities"])}
    train_tf, eval_tf = make_transforms(args.image_size)
    train_ds = TrainIdentityDataset(samples, split_indices["train"], train_label_to_class, train_tf)
    val_ds = EvalIdentityDataset(samples, split_indices["val"], eval_tf)
    test_ds = EvalIdentityDataset(samples, split_indices["test"], eval_tf)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model, public_weights = build_model(args.model, max(1, len(train_label_to_class)), args.random_weights)
    model = model.to(device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()

    config = StrictConfig(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        model=args.model,
        public_weights=public_weights,
        protocol=identity_splits["protocol"],
        hard_subsets=args.hard_subsets,
        seed=args.seed,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hard_val_fraction=args.hard_val_fraction,
        hard_test_fraction=args.hard_test_fraction,
        train_fractions=args.train_fractions,
        hard_train_fractions=args.hard_train_fractions,
        no_hard_train=args.no_hard_train,
        random_weights=args.random_weights,
        eval_random_only=args.eval_random_only,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "split_summary": summary,
                "identity_splits": identity_splits,
                "skipped_images": len(skipped),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "train_label_to_class.json").write_text(json.dumps(train_label_to_class, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "split_summary": summary}, indent=2))

    if args.eval_random_only:
        start = time.time()
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": 0,
                "config": asdict(config),
                "split_summary": summary,
                "identity_splits": identity_splits,
                "train_label_to_class": train_label_to_class,
            },
            args.output_dir / "random_model.pt",
        )
        val_data = collect_embeddings(model, val_loader, device, amp_enabled)
        val_verification, val_thresholds = verification_report(val_data)
        val_identification = identification_report(val_data)
        test_data = collect_embeddings(model, test_loader, device, amp_enabled)
        test_verification, _ = verification_report(test_data, val_thresholds=val_thresholds)
        test_identification = identification_report(test_data)
        metrics = {
            "best_epoch": None,
            "elapsed_seconds": round(time.time() - start, 3),
            "validation": {
                "verification": val_verification,
                "identification": val_identification,
            },
            "test": {
                "verification": test_verification,
                "identification": test_identification,
            },
            "split_summary": summary,
            "public_model": public_weights,
            "eval_random_only": True,
        }
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("\nFinal random-weight strict metrics")
        print(json.dumps({k: metrics[k] for k in ["validation", "test", "elapsed_seconds"]}, indent=2))
        return

    optimizer = make_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)

    best_val_auc = -1.0
    best_epoch = 0
    best_val_report = None
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp_enabled)
        scheduler.step()
        val_data = collect_embeddings(model, val_loader, device, amp_enabled)
        val_verification, val_thresholds = verification_report(val_data)
        val_identification = identification_report(val_data)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1_acc": train_metrics["top1_acc"],
            "val_auc": val_verification["auc"],
            "val_eer": val_verification["eer"],
            "val_best_balanced_acc": val_verification["thresholds"]["best_balanced"]["balanced_accuracy"],
            "val_identification_top1": val_identification["top1_acc"],
            "val_identification_top5": val_identification["top5_acc"],
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_verification["auc"] > best_val_auc:
            best_val_auc = val_verification["auc"]
            best_epoch = epoch
            best_val_report = {
                "verification": val_verification,
                "identification": val_identification,
                "thresholds": val_thresholds,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "config": asdict(config),
                    "split_summary": summary,
                    "identity_splits": identity_splits,
                    "train_label_to_class": train_label_to_class,
                    "val_report": best_val_report,
                },
                args.output_dir / "best_model.pt",
            )

    write_csv(args.output_dir / "history.csv", history)
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_data = collect_embeddings(model, val_loader, device, amp_enabled)
    val_verification, val_thresholds = verification_report(val_data)
    val_identification = identification_report(val_data)
    test_data = collect_embeddings(model, test_loader, device, amp_enabled)
    test_verification, _ = verification_report(test_data, val_thresholds=val_thresholds)
    test_identification = identification_report(test_data)
    metrics = {
        "best_epoch": best_epoch,
        "elapsed_seconds": round(time.time() - start, 3),
        "validation": {
            "verification": val_verification,
            "identification": val_identification,
        },
        "test": {
            "verification": test_verification,
            "identification": test_identification,
        },
        "split_summary": summary,
        "public_model": public_weights,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nFinal strict metrics")
    print(json.dumps({k: metrics[k] for k in ["best_epoch", "validation", "test", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
