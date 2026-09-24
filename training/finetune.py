#!/usr/bin/env python3
"""Fine-tune the Vietnamese Zipformer checkpoint on a CSV speech corpus.

Pipeline:
    CSV/audio
      -> Lhotse CutSet
      -> on-the-fly 80-bin FBank
      -> SentencePiece tokens
      -> Zipformer
      -> RNN-T + CTC + CR-CTC
      -> ScaledAdam + Eden

Typical usage:
    python training/finetune.py \
        --pretrained /path/to/pretrained.pt \
        --bpe-model /path/to/bpe.model \
        --train-csv data/processed/train.csv \
        --exp-dir checkpoints/finetune_v1
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import logging
import pathlib
import random
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import k2
import numpy as np
import sentencepiece as spm
import torch
from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.dataset import (
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    OnTheFlyFeatures,
    SpecAugment,
)
from lhotse.features import Fbank, FbankConfig
from torch.utils.data import DataLoader

from zipformer_model import (
    Eden,
    ScaledAdam,
    build_model_from_checkpoint_config,
    checkpoint_model_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------
# CLI and setup
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inputs = parser.add_argument_group("model and data")
    inputs.add_argument("--pretrained", required=True, help="Icefall training checkpoint")
    inputs.add_argument("--bpe-model", required=True, help="SentencePiece model")
    inputs.add_argument("--train-csv", required=True, help="Training CSV")
    inputs.add_argument("--valid-csv", help="Validation CSV; otherwise split train CSV")
    inputs.add_argument("--audio-root", help="Base directory for relative audio paths")
    inputs.add_argument("--audio-col", default="audio_path")
    inputs.add_argument("--text-col", default="label")

    training = parser.add_argument_group("training")
    training.add_argument("--exp-dir", default="checkpoints/finetune_v1")
    training.add_argument("--num-epochs", type=int, default=10)
    training.add_argument(
        "--base-lr",
        type=float,
        default=None,
        help="Fresh optimizer learning rate; checkpoint base_lr when omitted",
    )
    training.add_argument("--max-duration", type=float, default=100.0)
    training.add_argument("--valid-ratio", type=float, default=0.02)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--num-buckets", type=int, default=20)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--sample-rate", type=int, default=16000)
    runtime.add_argument("--num-workers", type=int, default=2)
    runtime.add_argument("--use-fp16", action="store_true")
    runtime.add_argument("--use-bf16", action="store_true")
    runtime.add_argument("--log-interval", type=int, default=20)
    runtime.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.use_fp16 and args.use_bf16:
        raise ValueError("Use only one of --use-fp16 and --use-bf16")
    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be positive")
    if args.max_duration <= 0:
        raise ValueError("--max-duration must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be positive")
    if args.sample_rate < 1:
        raise ValueError("--sample-rate must be positive")
    if args.base_lr is not None and args.base_lr <= 0:
        raise ValueError("--base-lr must be positive")
    if args.num_buckets is not None and args.num_buckets <= 0:
        raise ValueError("--num_buckets must be positive")

def project_path(value: str) -> Path:
    """Resolve project-relative CLI paths independently of the current working directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_paths(args: argparse.Namespace) -> dict[str, Optional[Path]]:
    paths = {
        "pretrained": project_path(args.pretrained),
        "bpe_model": project_path(args.bpe_model),
        "train_csv": project_path(args.train_csv),
        "valid_csv": project_path(args.valid_csv) if args.valid_csv else None,
        "audio_root": project_path(args.audio_root) if args.audio_root else None,
        "exp_dir": project_path(args.exp_dir),
    }

    for name in ("pretrained", "bpe_model", "train_csv"):
        path = paths[name]
        if path is None or not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} not found: {path}")

    valid_csv = paths["valid_csv"]
    if valid_csv is not None and not valid_csv.is_file():
        raise FileNotFoundError(f"--valid-csv not found: {valid_csv}")

    audio_root = paths["audio_root"]
    if audio_root is not None and not audio_root.is_dir():
        raise NotADirectoryError(f"--audio-root is not a directory: {audio_root}")

    return paths


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(args: argparse.Namespace) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_fp16 and device.type != "cuda":
        raise ValueError("--use-fp16 requires a CUDA device")
    if args.use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("--use-bf16 was requested, but this GPU does not support it")

    return device


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------


def read_csv_examples(
    csv_path: Path,
    audio_root: Optional[Path],
    audio_col: str,
    text_col: str,
) -> list[dict[str, Any]]:
    """Read and validate CSV rows, resolving each row to one utterance."""
    examples = []
    seen_rows = set()

    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        missing_columns = {audio_col, text_col} - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"CSV {csv_path} is missing columns: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            audio_value = (row.get(audio_col) or "").strip()
            transcript = (row.get(text_col) or "").strip()

            if not audio_value:
                raise ValueError(f"Empty audio path in {csv_path}, row {row_number}")
            if not transcript:
                raise ValueError(f"Empty transcript in {csv_path}, row {row_number}")

            audio_path = Path(audio_value).expanduser()
            if not audio_path.is_absolute():
                base = audio_root if audio_root is not None else csv_path.parent
                audio_path = base / audio_path
            audio_path = audio_path.resolve()

            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Audio file in {csv_path}, row {row_number} does not exist: "
                    f"{audio_path}"
                )

            row_key = (str(audio_path), transcript)
            if row_key in seen_rows:
                raise ValueError(
                    f"Duplicate audio/transcript row in {csv_path}, row {row_number}: "
                    f"{audio_path}"
                )
            seen_rows.add(row_key)

            digest = hashlib.sha1(
                f"{audio_path}\0{transcript}".encode("utf-8")
            ).hexdigest()[:20]
            examples.append(
                {
                    "id": f"utt-{digest}",
                    "audio_path": audio_path,
                    "transcript": transcript,
                    "row_key": row_key,
                }
            )

    if not examples:
        raise ValueError(f"CSV contains no utterances: {csv_path}")

    return examples


def prepare_train_valid_examples(
    train_examples: list[dict[str, Any]],
    valid_examples: Optional[list[dict[str, Any]]],
    valid_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use an explicit validation set or deterministically split the training CSV."""
    if valid_examples is not None:
        train_rows = {item["row_key"] for item in train_examples}
        valid_rows = {item["row_key"] for item in valid_examples}
        overlap = train_rows & valid_rows
        if overlap:
            path, transcript = next(iter(overlap))
            raise ValueError(
                "The same row occurs in training and validation data: "
                f"{path!r}, {transcript!r}"
            )
        return train_examples, valid_examples

    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be between 0 and 1")
    if len(train_examples) < 2:
        raise ValueError("At least two CSV rows are required for an automatic split")

    indices = list(range(len(train_examples)))
    random.Random(seed).shuffle(indices)

    valid_count = max(1, round(len(indices) * valid_ratio))
    valid_count = min(valid_count, len(indices) - 1)
    valid_indices = set(indices[:valid_count])

    train = [
        example
        for index, example in enumerate(train_examples)
        if index not in valid_indices
    ]
    valid = [
        example
        for index, example in enumerate(train_examples)
        if index in valid_indices
    ]
    return train, valid


def build_cutset_from_csv_rows(
    examples: list[dict[str, Any]],
    sample_rate: int,
) -> CutSet:
    """Build one full-utterance supervision for each CSV row."""
    recordings = []
    supervisions = []
    recording_ids = set()

    for example in examples:
        recording_id = example["id"]
        if recording_id in recording_ids:
            raise ValueError(f"Duplicate utterance ID: {recording_id}")
        recording_ids.add(recording_id)

        recording = Recording.from_file(
            example["audio_path"],
            recording_id=recording_id,
        )
        if recording.num_channels != 1:
            raise ValueError(
                f"Only mono audio is supported for now: {example['audio_path']} "
                f"has {recording.num_channels} channels"
            )
        if recording.sampling_rate != sample_rate:
            recording = recording.resample(sample_rate)

        recordings.append(recording)
        supervisions.append(
            SupervisionSegment(
                id=recording_id,
                recording_id=recording_id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                text=example["transcript"],
            )
        )

    return CutSet.from_manifests(
        recordings=RecordingSet.from_recordings(recordings),
        supervisions=SupervisionSet.from_segments(supervisions),
    )


def build_dataloader(
    cuts: CutSet,
    max_duration: float,
    num_workers: int,
    num_buckets: int,
    seed: int,
    shuffle: bool,
    sample_rate: int,
    feature_dim: int,
) -> DataLoader:
    feature_extractor = Fbank(
        FbankConfig(
            sampling_rate=sample_rate,
            num_mel_bins=feature_dim,
        )
    )
    dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(feature_extractor)
    )
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=max_duration,
        shuffle=shuffle,
        num_buckets=min(num_buckets, len(cuts)),
        seed=seed,
    )

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def prepare_data(
    args: argparse.Namespace,
    paths: Mapping[str, Optional[Path]],
    feature_dim: int,
) -> tuple[CutSet, CutSet, DataLoader, DataLoader]:
    train_examples = read_csv_examples(
        paths["train_csv"],
        paths["audio_root"],
        args.audio_col,
        args.text_col,
    )

    valid_examples = None
    if paths["valid_csv"] is not None:
        valid_examples = read_csv_examples(
            paths["valid_csv"],
            paths["audio_root"],
            args.audio_col,
            args.text_col,
        )

    train_examples, valid_examples = prepare_train_valid_examples(
        train_examples,
        valid_examples,
        args.valid_ratio,
        args.seed,
    )

    train_cuts = build_cutset_from_csv_rows(train_examples, args.sample_rate)
    valid_cuts = build_cutset_from_csv_rows(valid_examples, args.sample_rate)

    loader_options = {
        "max_duration": args.max_duration,
        "num_workers": args.num_workers,
        "num_bucket": args.num_buckets,
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "feature_dim": feature_dim,
    }
    train_loader = build_dataloader(train_cuts, shuffle=True, **loader_options)
    valid_loader = build_dataloader(valid_cuts, shuffle=False, **loader_options)

    return train_cuts, valid_cuts, train_loader, valid_loader


# ------------------------------------------------------------------
# Model and objective
# ------------------------------------------------------------------

def load_pretrained_checkpoint(path: Path) -> dict[str, Any]:
    with torch.serialization.safe_globals([pathlib.PosixPath]):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )

    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise TypeError("Expected an icefall checkpoint containing a 'model' state dict")

    return checkpoint


def load_tokenizer(path: Path, checkpoint: Mapping[str, Any]):
    tokenizer = spm.SentencePieceProcessor(model_file=str(path))

    expected_size = int(checkpoint["vocab_size"])
    if tokenizer.vocab_size() != expected_size:
        raise ValueError(
            f"SentencePiece vocabulary has {tokenizer.vocab_size()} entries, "
            f"but the checkpoint requires {expected_size}"
        )

    blank_id = int(checkpoint["blank_id"])
    blank_piece = tokenizer.id_to_piece(blank_id)
    if blank_piece != "<blk>":
        raise ValueError(
            f"Checkpoint blank_id={blank_id}, but that SentencePiece entry is "
            f"{blank_piece!r}, not '<blk>'"
        )

    return tokenizer


def checkpoint_loss_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "use_transducer",
        "use_ctc",
        "use_cr_ctc",
        "simple_loss_scale",
        "ctc_loss_scale",
        "cr_loss_scale",
        "prune_range",
        "am_scale",
        "lm_scale",
        "warm_step",
        "spec_aug_time_warp_factor",
        "time_mask_ratio",
        "subsampling_factor",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing loss configuration: {missing}")
    if checkpoint["use_cr_ctc"] and not checkpoint["use_ctc"]:
        raise ValueError("Checkpoint enables CR-CTC without enabling CTC")

    return {key: checkpoint[key] for key in required}


def build_spec_augment(loss_config: Mapping[str, Any]) -> SpecAugment:
    time_mask_ratio = float(loss_config["time_mask_ratio"])
    return SpecAugment(
        time_warp_factor=0,
        num_frame_masks=int(10 * time_mask_ratio),
        features_mask_size=27,
        num_feature_masks=2,
        frames_mask_size=100,
        max_frames_mask_fraction=0.15 * time_mask_ratio,
    )


def set_model_batch_count(model: torch.nn.Module, batch_count: float) -> None:
    """Refresh icefall ScheduledFloat modules with the current training progress."""
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def prepare_batch(
    model: torch.nn.Module,
    tokenizer,
    batch: dict[str, Any],
    use_spec_aug: bool,
) -> tuple[torch.Tensor, torch.Tensor, k2.RaggedTensor, list[list[int]], Any]:
    """Move one Lhotse batch to the model device and tokenize its transcripts."""
    device = next(model.parameters()).device
    features = batch["inputs"].to(device)
    if features.ndim != 3:
        raise ValueError(f"Expected features shaped (N, T, C), got {features.shape}")

    supervisions = batch["supervisions"]
    feature_lengths = supervisions["num_frames"].to(device)
    token_lists = tokenizer.encode(supervisions["text"], out_type=int)
    if any(not tokens for tokens in token_lists):
        raise ValueError("SentencePiece produced an empty token sequence")

    tokens = k2.RaggedTensor(token_lists)
    supervision_segments = None
    if use_spec_aug:
        supervision_segments = torch.stack(
            [
                supervisions["sequence_idx"],
                supervisions["start_frame"],
                supervisions["num_frames"],
            ],
            dim=1,
        )

    return features, feature_lengths, tokens, token_lists, supervision_segments


def combine_loss_terms(
    simple_loss: torch.Tensor,
    pruned_loss: torch.Tensor,
    ctc_loss: torch.Tensor,
    cr_loss: torch.Tensor,
    loss_config: Mapping[str, Any],
    global_step: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the warmups and scales used by the current icefall recipe."""
    warm_step = int(loss_config["warm_step"])
    if warm_step <= 0:
        raise ValueError(f"warm_step must be positive, got {warm_step}")

    total_loss = torch.zeros((), device=device)
    stats = {}

    if loss_config["use_transducer"]:
        target_scale = float(loss_config["simple_loss_scale"])
        progress = min(global_step / warm_step, 1.0)
        simple_scale = 1.0 - progress * (1.0 - target_scale)
        pruned_scale = 0.1 + 0.9 * progress

        total_loss = total_loss + simple_scale * simple_loss
        total_loss = total_loss + pruned_scale * pruned_loss
        stats.update(
            simple_loss=simple_loss.detach().item(),
            pruned_loss=pruned_loss.detach().item(),
            simple_loss_scale=simple_scale,
            pruned_loss_scale=pruned_scale,
        )

    if loss_config["use_ctc"]:
        ctc_scale = float(loss_config["ctc_loss_scale"])
        total_loss = total_loss + ctc_scale * ctc_loss
        stats.update(
            ctc_loss=ctc_loss.detach().item(),
            ctc_loss_scale=ctc_scale,
        )

        if loss_config["use_cr_ctc"]:
            cr_scale = min(global_step / warm_step, 1.0) * float(
                loss_config["cr_loss_scale"]
            )
            total_loss = total_loss + cr_scale * cr_loss
            stats.update(
                cr_loss=cr_loss.detach().item(),
                cr_loss_scale=cr_scale,
            )

    return total_loss, stats


def compute_loss(
    model: torch.nn.Module,
    tokenizer,
    batch: dict[str, Any],
    loss_config: Mapping[str, Any],
    global_step: int,
    is_training: bool,
    spec_augment: Optional[SpecAugment],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run Zipformer forward and combine RNN-T, CTC, and CR-CTC objectives."""
    use_cr_ctc = bool(loss_config["use_cr_ctc"])
    use_spec_aug = use_cr_ctc and is_training

    features, feature_lengths, tokens, token_lists, supervision_segments = prepare_batch(
        model,
        tokenizer,
        batch,
        use_spec_aug,
    )
    device = features.device

    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss, ctc_loss, _, cr_loss = model(
            x=features,
            x_lens=feature_lengths,
            y=tokens,
            prune_range=int(loss_config["prune_range"]),
            am_scale=float(loss_config["am_scale"]),
            lm_scale=float(loss_config["lm_scale"]),
            use_cr_ctc=use_cr_ctc,
            use_spec_aug=use_spec_aug,
            spec_augment=spec_augment,
            supervision_segments=supervision_segments,
            time_warp_factor=int(loss_config["spec_aug_time_warp_factor"]),
        )

        total_loss, loss_stats = combine_loss_terms(
            simple_loss,
            pruned_loss,
            ctc_loss,
            cr_loss,
            loss_config,
            global_step,
            device,
        )

    stats: dict[str, Any] = {
        "feature_shape": tuple(features.shape),
        "feature_lengths": feature_lengths.detach().cpu().tolist(),
        "token_lengths": [len(tokens) for tokens in token_lists],
        "frames": int(
            (
                feature_lengths // int(loss_config["subsampling_factor"])
            ).sum().item()
        ),
        **loss_stats,
        "loss": total_loss.detach().item(),
    }

    ensure_finite_loss(total_loss, stats)
    return total_loss, stats


def ensure_finite_loss(total_loss: torch.Tensor, stats: Mapping[str, Any]) -> None:
    if not torch.isfinite(total_loss):
        raise FloatingPointError(f"Non-finite total loss: {dict(stats)}")

    for name in ("simple_loss", "pruned_loss", "ctc_loss", "cr_loss"):
        if name in stats and not np.isfinite(stats[name]):
            raise FloatingPointError(f"Non-finite {name}: {stats[name]}")


def autocast_context(device: torch.device, args: argparse.Namespace):
    if args.use_fp16:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    if args.use_bf16:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    tokenizer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Eden,
    scaler,
    loss_config: Mapping[str, Any],
    spec_augment: Optional[SpecAugment],
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    train_loader.sampler.set_epoch(epoch - 1)
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_frames = 0

    for batch_index, batch in enumerate(train_loader, start=1):
        # Upstream refreshes ScheduledFloat values every ten batches. Keeping
        # that cadence avoids changing its layer-drop/dropout schedule.
        if (batch_index - 1) % 10 == 0:
            set_model_batch_count(model, global_step * args.max_duration / 600.0)

        global_step += 1
        with autocast_context(device, args):
            loss, stats = compute_loss(
                model,
                tokenizer,
                batch,
                loss_config,
                global_step,
                is_training=True,
                spec_augment=spec_augment,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scheduler.step_batch(global_step)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            scheduler.step_batch(global_step)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += stats["loss"]
        total_frames += stats["frames"]

        if batch_index % args.log_interval == 0:
            logging.info(
                "epoch=%d batch=%d step=%d loss/frame=%.4f lr=%.3g",
                epoch,
                batch_index,
                global_step,
                stats["loss"] / max(stats["frames"], 1),
                scheduler.get_last_lr()[0],
            )

    return total_loss / max(total_frames, 1), global_step


def validate(
    model: torch.nn.Module,
    tokenizer,
    valid_loader: DataLoader,
    loss_config: Mapping[str, Any],
    spec_augment: Optional[SpecAugment],
    args: argparse.Namespace,
    device: torch.device,
    global_step: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_frames = 0

    for batch in valid_loader:
        with autocast_context(device, args):
            _, stats = compute_loss(
                model,
                tokenizer,
                batch,
                loss_config,
                global_step,
                is_training=False,
                spec_augment=spec_augment,
            )
        total_loss += stats["loss"]
        total_frames += stats["frames"]

    return total_loss / max(total_frames, 1)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Eden,
    scaler,
    epoch: int,
    global_step: int,
    validation_loss: float,
    args: argparse.Namespace,
    model_config: Mapping[str, Any],
    loss_config: Mapping[str, Any],
) -> None:
    training_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "grad_scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "validation_loss": validation_loss,
            "training_arguments": training_arguments,
            "model_config": dict(model_config),
            "loss_config": dict(loss_config),
        },
        path,
    )


def run_dry_run(
    model: torch.nn.Module,
    tokenizer,
    train_loader: DataLoader,
    loss_config: Mapping[str, Any],
    spec_augment: Optional[SpecAugment],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    model.train()
    batch = next(iter(train_loader))
    set_model_batch_count(model, 0.0)

    with autocast_context(device, args):
        _, stats = compute_loss(
            model,
            tokenizer,
            batch,
            loss_config,
            global_step=1,
            is_training=True,
            spec_augment=spec_augment,
        )

    logging.info("Dry-run feature shape: %s", stats["feature_shape"])
    logging.info("Dry-run feature lengths: %s", stats["feature_lengths"])
    logging.info("Dry-run token lengths: %s", stats["token_lengths"])
    for name in ("simple_loss", "pruned_loss", "ctc_loss", "cr_loss", "loss"):
        if name in stats:
            logging.info("Dry-run %s: %.6f", name, stats[name])
    logging.info("DRY RUN: PASS (no optimizer step was performed)")


# ------------------------------------------------------------------
# Logging and entry point
# ------------------------------------------------------------------


def log_startup(
    paths: Mapping[str, Optional[Path]],
    train_cuts: CutSet,
    valid_cuts: CutSet,
    parameter_count: int,
    trainable_count: int,
    device: torch.device,
    loss_config: Mapping[str, Any],
) -> None:
    train_hours = sum(cut.duration for cut in train_cuts) / 3600
    valid_hours = sum(cut.duration for cut in valid_cuts) / 3600

    logging.info("Pretrained checkpoint: %s", paths["pretrained"])
    logging.info("BPE model: %s", paths["bpe_model"])
    logging.info("Train: %d utterances, %.2f hours", len(train_cuts), train_hours)
    logging.info(
        "Validation: %d utterances, %.2f hours",
        len(valid_cuts),
        valid_hours,
    )
    logging.info(
        "Parameters: %s total, %s trainable",
        f"{parameter_count:,}",
        f"{trainable_count:,}",
    )
    logging.info("Device: %s", device)
    logging.info(
        "Loss scales: simple=%s ctc=%s cr=%s",
        loss_config["simple_loss_scale"],
        loss_config["ctc_loss_scale"],
        loss_config["cr_loss_scale"],
    )
    logging.info("Experiment directory: %s", paths["exp_dir"])


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    validate_args(args)
    seed_everything(args.seed)

    # 1. Resolve inputs and recover the checkpoint configuration.
    paths = resolve_paths(args)
    checkpoint = load_pretrained_checkpoint(paths["pretrained"])
    tokenizer = load_tokenizer(paths["bpe_model"], checkpoint)
    model_config = checkpoint_model_config(checkpoint)
    loss_config = checkpoint_loss_config(checkpoint)

    # 2. Build the data pipeline using the checkpoint's acoustic feature size.
    train_cuts, valid_cuts, train_loader, valid_loader = prepare_data(
        args,
        paths,
        feature_dim=int(checkpoint["feature_dim"]),
    )

    # 3. Reconstruct Zipformer and require an exact pretrained weight match.
    model = build_model_from_checkpoint_config(checkpoint)
    strict_result = model.load_state_dict(checkpoint["model"], strict=True)
    if strict_result.missing_keys or strict_result.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {strict_result}")
    logging.info(
        "Strict checkpoint load: PASS (%d state entries)",
        len(checkpoint["model"]),
    )

    base_lr = float(args.base_lr if args.base_lr is not None else checkpoint["base_lr"])
    lr_batches = float(checkpoint["lr_batches"])
    lr_epochs = float(checkpoint["lr_epochs"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    del checkpoint
    gc.collect()

    # 4. Move the model to the selected device and configure augmentation.
    device = select_device(args)
    model.to(device)
    spec_augment = (
        build_spec_augment(loss_config)
        if loss_config["use_cr_ctc"]
        else None
    )

    log_startup(
        paths,
        train_cuts,
        valid_cuts,
        parameter_count,
        trainable_count,
        device,
        loss_config,
    )

    if args.dry_run:
        run_dry_run(
            model,
            tokenizer,
            train_loader,
            loss_config,
            spec_augment,
            args,
            device,
        )
        return 0

    # 5. Fine-tune with fresh optimizer/scheduler state.
    paths["exp_dir"].mkdir(parents=True, exist_ok=True)
    optimizer = ScaledAdam(
        model.named_parameters(),
        lr=base_lr,
        clipping_scale=2.0,
    )
    scheduler = Eden(
        optimizer,
        lr_batches=lr_batches,
        lr_epochs=lr_epochs,
        warmup_start=0.1,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=args.use_fp16 or args.use_bf16,
        init_scale=1.0,
    )

    global_step = 0
    best_validation_loss = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        seed_everything(args.seed + epoch - 1)
        scheduler.step_epoch(epoch - 1)

        training_loss, global_step = train_one_epoch(
            model,
            tokenizer,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            loss_config,
            spec_augment,
            args,
            device,
            epoch,
            global_step,
        )
        validation_loss = validate(
            model,
            tokenizer,
            valid_loader,
            loss_config,
            spec_augment,
            args,
            device,
            global_step,
        )

        logging.info(
            "epoch=%d train_loss/frame=%.5f valid_loss/frame=%.5f",
            epoch,
            training_loss,
            validation_loss,
        )

        epoch_path = paths["exp_dir"] / f"epoch-{epoch}.pt"
        save_checkpoint(
            epoch_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            validation_loss,
            args,
            model_config,
            loss_config,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            shutil.copyfile(epoch_path, paths["exp_dir"] / "best-valid.pt")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        if "--dry-run" in sys.argv:
            logging.exception("DRY RUN: FAIL")
            raise SystemExit(1)
        raise
