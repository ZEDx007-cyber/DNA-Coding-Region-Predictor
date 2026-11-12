import tempfile
import unittest
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from dna_predictor import (
    CODONS,
    FEATURE_NAMES,
    extract_features,
    labels_for_records,
    load_labeled_fasta,
    train_model,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_features_include_gc_and_normalized_frame_zero_codons(self):
        features = extract_features("ATGGCC")

        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertAlmostEqual(features[0], 4 / 6)
        self.assertAlmostEqual(features[1 + CODONS.index("ATG")], 0.5)
        self.assertAlmostEqual(features[1 + CODONS.index("GCC")], 0.5)
        self.assertAlmostEqual(sum(features[1:]), 1.0)

    def test_ambiguous_bases_are_ignored_and_rna_is_accepted(self):
        features = extract_features("AUGNNNGCC")

        self.assertAlmostEqual(features[0], 4 / 6)
        self.assertAlmostEqual(features[1 + CODONS.index("ATG")], 0.5)
        self.assertAlmostEqual(features[1 + CODONS.index("GCC")], 0.5)

    def test_rejects_sequences_without_usable_features(self):
        for sequence in ("NNN", "A"):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                extract_features(sequence)


class FastaAndModelTests(unittest.TestCase):
    def test_labeled_fasta_accepts_a_utf8_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "bom.fasta"
            fasta.write_text(">sample label=coding\nATG\n", encoding="utf-8-sig")

            records, labels = load_labeled_fasta(fasta)

        self.assertEqual(len(records), 1)
        self.assertEqual(labels, ["coding"])

    def test_labeled_fasta_and_training(self):
        records = []
        labels = []
        for label, sequence in (
            ("coding", "ATGGCCGCT"),
            ("coding", "ATGGCCGCC"),
            ("coding", "ATGGCTGCC"),
            ("noncoding", "ATATATATA"),
            ("noncoding", "TATATATAT"),
            ("noncoding", "ATATTTATA"),
        ):
            record = SeqRecord(
                Seq(sequence), id=f"sample{len(records)}|label={label}", description=""
            )
            records.append(record)
            labels.append(label)

        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "labeled.fasta"
            SeqIO.write(records, fasta, "fasta")
            loaded_records, loaded_labels = load_labeled_fasta(fasta)

        self.assertEqual(loaded_labels, labels)
        model, x_test, y_test = train_model(loaded_records, loaded_labels)
        self.assertEqual(len(model.predict(x_test)), len(y_test))

    def test_labeled_fasta_requires_a_known_label(self):
        record = SeqRecord(Seq("ATG"), id="unlabeled", description="")
        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "unlabeled.fasta"
            SeqIO.write([record], fasta, "fasta")
            with self.assertRaisesRegex(ValueError, "must include"):
                load_labeled_fasta(fasta)

    def test_labels_for_records_rejects_empty_or_unlabeled_records(self):
        with self.assertRaisesRegex(ValueError, "No FASTA records"):
            labels_for_records([])

        record = SeqRecord(Seq("ATG"), id="unlabeled", description="")
        with self.assertRaisesRegex(ValueError, "must include"):
            labels_for_records([record])

    def test_curated_demo_fasta_is_balanced_and_trainable(self):
        fasta = Path(__file__).parent / "data" / "lambda_labeled.fasta"
        records, labels = load_labeled_fasta(fasta)

        self.assertEqual(len(records), 36)
        self.assertEqual(labels.count("coding"), 18)
        self.assertEqual(labels.count("noncoding"), 18)
        model, x_test, y_test = train_model(records, labels)
        self.assertEqual(len(model.predict(x_test)), len(y_test))


if __name__ == "__main__":
    unittest.main()
