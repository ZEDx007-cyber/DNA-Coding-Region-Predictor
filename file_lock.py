"""Passphrase-based encryption for private local DNA Studio files."""

from __future__ import annotations

import argparse
import base64
import getpass
import secrets
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


MAGIC = b"DNAENC1"
SALT_LENGTH = 16
KDF_ITERATIONS = 600_000
MIN_PASSPHRASE_LENGTH = 12


def _fernet(password: str, salt: bytes) -> Fernet:
    if len(password) < MIN_PASSPHRASE_LENGTH:
        raise ValueError(
            f"Passphrase must contain at least {MIN_PASSPHRASE_LENGTH} characters"
        )
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=KDF_ITERATIONS,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
    return Fernet(key)


def _distinct_paths(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output paths must be different")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")


def encrypt_file(source: Path, destination: Path, password: str) -> None:
    """Encrypt a local file, refusing to overwrite the input or an existing output."""
    _distinct_paths(source, destination)
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    salt = secrets.token_bytes(SALT_LENGTH)
    ciphertext = MAGIC + salt + _fernet(password, salt).encrypt(source.read_bytes())
    with destination.open("xb") as output:
        output.write(ciphertext)


def decrypt_file(source: Path, destination: Path, password: str) -> None:
    """Decrypt a DNA Studio encrypted file to a new local file."""
    _distinct_paths(source, destination)
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    encrypted = source.read_bytes()
    header_length = len(MAGIC) + SALT_LENGTH
    if len(encrypted) <= header_length or not encrypted.startswith(MAGIC):
        raise ValueError("File is not a supported DNA Studio encrypted file")
    salt = encrypted[len(MAGIC) : header_length]
    try:
        plaintext = _fernet(password, salt).decrypt(encrypted[header_length:])
    except InvalidToken as error:
        raise ValueError("Wrong passphrase or encrypted file is damaged") from error
    with destination.open("xb") as output:
        output.write(plaintext)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt or decrypt a private local file.")
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("encrypt", "decrypt"):
        command = commands.add_parser(action)
        command.add_argument("input", type=Path)
        command.add_argument("output", type=Path)
    args = parser.parse_args()

    password = getpass.getpass("Passphrase: ")
    if args.command == "encrypt":
        confirmation = getpass.getpass("Confirm passphrase: ")
        if password != confirmation:
            parser.error("Passphrases do not match")
    try:
        if args.command == "encrypt":
            encrypt_file(args.input, args.output, password)
        else:
            decrypt_file(args.input, args.output, password)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
