"""Predict coding status from GC content and in-frame codon frequencies."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from typing import TextIO

import joblib
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


BASES = "ACGT"
CODONS = tuple("".join(codon) for codon in product(BASES, repeat=3))
FEATURE_NAMES = ("gc_content", *CODONS)
LABELS = ("coding", "noncoding")
LABEL_PATTERN = re.compile(r"label\s*=\s*(coding|non[-_ ]?coding)\b", re.IGNORECASE)


def extract_features(sequence: str) -> np.ndarray:
    """Return GC fraction followed by 64 frame-0 codon frequencies."""
    sequence = "".join(sequence.split()).upper().replace("U", "T")
    valid_bases = [base for base in sequence if base in BASES]
    if not valid_bases:
        raise ValueError("sequence contains no unambiguous DNA bases")

    codon_counts = Counter(
        sequence[index : index + 3]
        for index in range(0, len(sequence) - 2, 3)
        if sequence[index : index + 3] in CODONS
    )
    codon_total = sum(codon_counts.values())
    if codon_total == 0:
        raise ValueError("sequence contains no unambiguous codons in reading frame 0")

    gc_content = sum(base in "GC" for base in valid_bases) / len(valid_bases)
    return np.asarray(
        [gc_content, *(codon_counts[codon] / codon_total for codon in CODONS)],
        dtype=float,
    )


def _read_fasta(path: Path) -> list[SeqRecord]:
    with path.open(encoding="utf-8-sig") as fasta_file:
        records = list(SeqIO.parse(fasta_file, "fasta"))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def labels_for_records(records: list[SeqRecord]) -> list[str]:
    """Return labels from FASTA record descriptions."""
    if not records:
        raise ValueError("No FASTA records found")
    labels: list[str] = []
    for record in records:
        match = LABEL_PATTERN.search(record.description)
        if match is None:
            raise ValueError(
                f"FASTA record {record.id!r} must include "
                "'label=coding' or 'label=noncoding' in its header"
            )
        label = match.group(1).lower().replace("-", "").replace("_", "").replace(" ", "")
        labels.append("coding" if label == "coding" else "noncoding")
    return labels


def load_labeled_fasta(path: Path) -> tuple[list[SeqRecord], list[str]]:
    """Read FASTA records whose descriptions include label=coding|noncoding."""
    records = _read_fasta(path)
    labels = labels_for_records(records)
    return records, labels


def _feature_matrix(records: list[SeqRecord]) -> np.ndarray:
    rows = []
    for record in records:
        try:
            rows.append(extract_features(str(record.seq)))
        except ValueError as error:
            raise ValueError(f"FASTA record {record.id!r}: {error}") from error
    return np.vstack(rows)


def train_model(
    records: list[SeqRecord],
    labels: list[str],
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    """Fit a standardized logistic-regression model and return its held-out split."""
    if not 0 < test_fraction <= 0.5:
        raise ValueError("test fraction must be greater than 0 and at most 0.5")
    counts = Counter(labels)
    if any(counts[label] < 2 for label in LABELS):
        raise ValueError("Training requires at least two FASTA records per label")

    features = _feature_matrix(records)
    test_count = max(len(LABELS), math.ceil(len(labels) * test_fraction))
    if test_count > len(labels) - len(LABELS):
        raise ValueError("Not enough records for a stratified train/test split")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        np.asarray(labels),
        test_size=test_count,
        random_state=random_state,
        stratify=labels,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
    )
    model.fit(x_train, y_train)
    return model, x_test, y_test


def _train_command(args: argparse.Namespace) -> None:
    records, labels = load_labeled_fasta(args.fasta)
    model, x_test, y_test = train_model(
        records, labels, test_fraction=args.test_fraction, random_state=args.seed
    )
    predictions = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(
        classification_report(
            y_test, predictions, labels=list(LABELS), target_names=list(LABELS), zero_division=0
        )
    )

    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "labels": LABELS,
    }
    joblib.dump(bundle, args.model)
    print(f"Model saved to {args.model}")


def _predict_command(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model)
    if (
        not isinstance(bundle, dict)
        or bundle.get("feature_names") != FEATURE_NAMES
        or not {"model", "labels"}.issubset(bundle)
    ):
        raise ValueError("Model file is incompatible with this version of the predictor")

    records = _read_fasta(args.fasta)
    features = _feature_matrix(records)
    model = bundle["model"]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    class_indices = {label: index for index, label in enumerate(model.classes_)}

    output: TextIO
    context = nullcontext(sys.stdout) if str(args.output) == "-" else args.output.open(
        "w", newline="", encoding="utf-8"
    )
    with context as output:
        writer = csv.writer(output)
        writer.writerow(("sequence_id", "predicted_label", "coding_probability"))
        for record, prediction, probability in zip(records, predictions, probabilities):
            writer.writerow(
                (
                    record.id,
                    prediction,
                    f"{probability[class_indices['coding']]:.6f}",
                )
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict coding/noncoding DNA from GC content and codon frequencies."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train and evaluate using labeled FASTA")
    train_parser.add_argument("fasta", type=Path, help="FASTA with label=... in each header")
    train_parser.add_argument("--model", type=Path, default=Path("dna_model.joblib"))
    train_parser.add_argument("--test-fraction", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(run=_train_command)

    predict_parser = commands.add_parser("predict", help="predict labels for FASTA records")
    predict_parser.add_argument("fasta", type=Path, help="FASTA file to classify")
    predict_parser.add_argument("--model", type=Path, default=Path("dna_model.joblib"))
    predict_parser.add_argument(
        "--output", type=Path, default=Path("-"), help="CSV output path, or - for stdout"
    )
    predict_parser.set_defaults(run=_predict_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()