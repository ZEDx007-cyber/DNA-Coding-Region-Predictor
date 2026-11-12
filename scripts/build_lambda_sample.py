"""Rebuild the balanced FASTA sample from NCBI RefSeq NC_001416.1."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


ACCESSION = "NC_001416.1"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ANNOTATED_GENE_TYPES = {"CDS", "tRNA", "rRNA", "ncRNA", "tmRNA"}


def fetch_reference(email: str = ""):
    params = {
        "db": "nuccore",
        "id": ACCESSION,
        "rettype": "gbwithparts",
        "retmode": "text",
        "tool": "DNAStudio",
    }
    if email:
        params["email"] = email
    request = Request(
        f"{NCBI_EFETCH}?{urlencode(params)}",
        headers={"User-Agent": "DNAStudio/1.0", "Accept": "text/plain"},
    )
    with urlopen(request, timeout=60) as response:
        data = response.read().decode("utf-8")
    return SeqIO.read(io.StringIO(data), "genbank")


def _evenly_sample(items: list[tuple], count: int) -> list[tuple]:
    return [items[min(len(items) - 1, int(index * len(items) / count))] for index in range(count)]


def build_records(record, sample_count: int = 18) -> list[SeqRecord]:
    coding = []
    for feature in record.features:
        if feature.type != "CDS":
            continue
        sequence = str(feature.extract(record.seq)).upper().replace("U", "T")
        if 90 <= len(sequence) <= 3000 and set(sequence) <= set("ACGT"):
            coding.append(
                (
                    int(feature.location.start),
                    int(feature.location.end),
                    sequence,
                    feature.qualifiers.get("product", ["CDS"])[0],
                )
            )

    covered = sorted(
        (int(feature.location.start), int(feature.location.end))
        for feature in record.features
        if feature.type in ANNOTATED_GENE_TYPES
    )
    intergenic = []
    covered_end = 0
    for start, end in covered:
        gap_length = start - covered_end
        if 90 <= gap_length <= 3000:
            sequence = str(record.seq[covered_end:start]).upper().replace("U", "T")
            if set(sequence) <= set("ACGT"):
                intergenic.append((covered_end, start, sequence))
        covered_end = max(covered_end, end)

    count = min(sample_count, len(coding), len(intergenic))
    if count < 2:
        raise ValueError(
            f"Reference contains too few usable examples: "
            f"{len(coding)} coding and {len(intergenic)} intergenic"
        )

    records = []
    for index, (start, end, sequence, product) in enumerate(
        _evenly_sample(coding, count), 1
    ):
        records.append(
            SeqRecord(
                Seq(sequence),
                id=f"lambda_coding_{index}",
                description=(
                    f"label=coding source={ACCESSION} coordinates={start + 1}-{end} "
                    f"product={product}"
                ),
            )
        )
    for index, (start, end, sequence) in enumerate(
        _evenly_sample(intergenic, count), 1
    ):
        records.append(
            SeqRecord(
                Seq(sequence),
                id=f"lambda_noncoding_{index}",
                description=(
                    f"label=noncoding source={ACCESSION} "
                    f"coordinates={start + 1}-{end}"
                ),
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", default="", help="optional contact email for NCBI")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "lambda_labeled.fasta",
    )
    args = parser.parse_args()
    reference = fetch_reference(args.email.strip())
    records = build_records(reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, args.output, "fasta")
    print(
        f"Wrote {len(records)} examples from {reference.id} "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
