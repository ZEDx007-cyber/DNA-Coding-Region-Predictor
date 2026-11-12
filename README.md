# DNA Studio

DNA Studio is a small educational app for searching public nucleotide records,
training a coding/noncoding baseline from labeled FASTA, and predicting
sequence labels. It includes a Windows desktop interface, a browser interface,
and a command-line workflow.

**Copyright © 2026 Ishan Kundu. All rights reserved.** See [LICENSE](./LICENSE).

## Features

- Search NCBI Nucleotide and ENA; download selected public records as FASTA.
- Train a session-local logistic-regression model from labeled FASTA.
- Predict coding probabilities and export results as CSV.
- Start the Windows app or run the web UI locally.
- Encrypt private local files with a passphrase entered at runtime.

## Install

Use Python 3.12 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run locally

Windows desktop:

```powershell
python dna_app.py
```

Browser interface:

```powershell
streamlit run web_app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`.

The web app starts with a demo model trained in memory from the bundled
reference FASTA. Visitors can upload their own labeled FASTA and train a model
in their browser session. Custom models, uploaded records, and prediction
results are not saved to shared storage or shared between visitors. Archive
records are not automatically labeled; supervised training requires
`label=coding` or `label=noncoding` in every FASTA header.

## Run in Docker

```powershell
docker build -t dna-studio .
docker run --rm -p 8501:8501 dna-studio
```

Then open `http://localhost:8501`. `render.yaml` and the Dockerfile provide a
starting point for a Docker-capable host. Creating an account, connecting a
repository, choosing a paid/free plan, and publishing the service must be done
by the owner on the hosting provider. This workspace has no connected
publishing or deployment account, so the app has **not** been published online.

Before exposing the app publicly, review the host's privacy and resource
policies. Visitors can upload up to 20 MiB, 2,500 records, and 100,000 bases per
record. Training runs within the visitor's server-side session and is not
persistent; this is not a shared community-training service.

## Demo data and model

The demo data is a balanced sample of 18 annotated coding sequences and 18
intergenic gaps derived from the NCBI RefSeq / GenBank bacteriophage lambda
record **NC_001416.1**. Details and limits are documented in
[data/README.md](./data/README.md). The web app trains its demo model from that
sample at startup; it does not load user-uploaded model files.

The predictor uses GC content and frame-0 codon frequencies. This small
single-reference example is for demonstration only. It is not a gene caller or
a validated biological diagnostic, and its held-out score should not be
interpreted as evidence of performance on other organisms.

Archive integrations use the [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
and [ENA Portal / Browser APIs](https://ena-docs.readthedocs.io/en/latest/retrieval/programmatic-access.html).
NCBI contact email is optional. WHO is not a general public nucleotide FASTA
archive, so the app does not label WHO surveillance data as sequence downloads.

## Encrypt a private local file

The public sample data and source code remain readable so the app can run.
To protect a **private local** FASTA or model file, use the encryption helper:

```powershell
python file_lock.py encrypt private.fasta private.fasta.enc
python file_lock.py decrypt private.fasta.enc recovered.fasta
```

The helper prompts for a passphrase and does not store it in source, the
repository, or the encrypted file. Use a new strong passphrase; do not reuse a
password that has been posted in a chat. Encrypted files (`*.enc`) are excluded
from Git by default.

## Data safety

Do not upload sensitive or personally identifiable sequence data to a public
deployment unless you have permission and have reviewed that host's data
handling terms. The repository excludes `.env`, local databases, model files,
virtual environments, and encrypted files from Git.

## Tests

```powershell
python -m unittest -v
```

## License and ownership

The original source code and documentation are proprietary and all rights are
reserved by Ishan Kundu. No reuse, redistribution, or derivative-work
permission is granted by this repository. Third-party dependencies and the
NCBI-derived reference data retain their own applicable terms; see
[LICENSE](./LICENSE) and [data/README.md](./data/README.md).
