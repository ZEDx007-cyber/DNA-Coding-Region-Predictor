import unittest
import io
from unittest.mock import patch

from sequence_sources import (
    MAX_RECORD_DOWNLOAD_BYTES,
    SequenceHit,
    _read_url,
    download_fasta,
    search_sequences,
)


class SequenceSourceTests(unittest.TestCase):
    @patch("sequence_sources._read_url")
    def test_ncbi_search_uses_summaries_and_contact_email(self, read_url):
        read_url.side_effect = [
            '{"esearchresult":{"idlist":["123","456"]}}',
            (
                '{"result":{"uids":["123","456"],'
                '"123":{"accessionversion":"AB123.1","title":"First record"},'
                '"456":{"accessionversion":"CD456.1","title":"Second record"}}}'
            ),
        ]

        hits = search_sequences("ncbi", "example", limit=2, email="researcher@example.org")

        self.assertEqual(
            hits,
            [
                SequenceHit("ncbi", "AB123.1", "First record"),
                SequenceHit("ncbi", "CD456.1", "Second record"),
            ],
        )
        self.assertEqual(read_url.call_count, 2)

    @patch("sequence_sources._read_url")
    def test_ena_search_parses_tsv_rows(self, read_url):
        read_url.return_value = (
            "accession\tdescription\n"
            "AB123.1\tExample gene\n"
            "CD456.1\tAnother gene\n"
        )

        hits = search_sequences("ena", "example gene", limit=10)

        self.assertEqual(
            hits,
            [
                SequenceHit("ena", "AB123.1", "Example gene"),
                SequenceHit("ena", "CD456.1", "Another gene"),
            ],
        )
        self.assertIn('description="*example gene*"', read_url.call_args.args[1]["query"])

    @patch("sequence_sources._read_url")
    def test_download_rejects_non_fasta_archive_responses(self, read_url):
        read_url.return_value = "not a FASTA record"

        with self.assertRaisesRegex(ValueError, "did not contain FASTA"):
            download_fasta(SequenceHit("ena", "AB123.1", "Example"))

    @patch("sequence_sources._read_url")
    def test_download_returns_normalized_fasta(self, read_url):
        read_url.return_value = ">AB123.1 example\nATGC"

        fasta = download_fasta(SequenceHit("ncbi", "AB123.1", "Example"))

        self.assertEqual(fasta, ">AB123.1 example\nATGC\n")
        self.assertIn("efetch.fcgi", read_url.call_args.args[0])

    @patch("sequence_sources._read_url")
    def test_ena_download_uses_browser_fasta_endpoint(self, read_url):
        read_url.return_value = ">ENA|AB123.1|Example\nATGC\n"

        download_fasta(SequenceHit("ena", "AB123.1", "Example"))

        self.assertEqual(
            read_url.call_args.args[0],
            "https://www.ebi.ac.uk/ena/browser/api/fasta/AB123.1",
        )
        self.assertEqual(read_url.call_args.args[1], {})
        self.assertEqual(
            read_url.call_args.kwargs["max_bytes"], MAX_RECORD_DOWNLOAD_BYTES
        )

    @patch("sequence_sources.urlopen")
    def test_read_url_rejects_responses_over_the_configured_limit(self, urlopen_mock):
        urlopen_mock.return_value = io.BytesIO(b"too large")

        with self.assertRaisesRegex(ValueError, "download limit"):
            _read_url("https://example.test/record", {}, max_bytes=4)


if __name__ == "__main__":
    unittest.main()
