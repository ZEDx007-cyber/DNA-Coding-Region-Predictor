import tempfile
import unittest
from pathlib import Path

from file_lock import decrypt_file, encrypt_file


class FileLockTests(unittest.TestCase):
    def test_encrypt_and_decrypt_round_trip(self):
        payload = b">record label=coding\nATGGCCGCT\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "training.fasta"
            encrypted = root / "training.fasta.enc"
            decrypted = root / "recovered.fasta"
            source.write_bytes(payload)

            encrypt_file(source, encrypted, "new private passphrase")
            decrypt_file(encrypted, decrypted, "new private passphrase")

            self.assertNotIn(payload, encrypted.read_bytes())
            self.assertEqual(decrypted.read_bytes(), payload)

    def test_decryption_rejects_wrong_passphrase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "private.txt"
            encrypted = root / "private.txt.enc"
            source.write_text("private", encoding="utf-8")
            encrypt_file(source, encrypted, "new private passphrase")

            with self.assertRaisesRegex(ValueError, "Wrong passphrase"):
                decrypt_file(encrypted, root / "recovered.txt", "a different passphrase")

    def test_encryption_rejects_short_passphrase_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "private.txt"
            destination = root / "private.txt.enc"
            source.write_text("private", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "at least 12"):
                encrypt_file(source, destination, "too short")

            encrypt_file(source, destination, "new private passphrase")
            with self.assertRaises(FileExistsError):
                encrypt_file(source, destination, "new private passphrase")


if __name__ == "__main__":
    unittest.main()
