"""Public-facing Streamlit interface for DNA Studio."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import streamlit as st
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from sklearn.metrics import accuracy_score, classification_report

from dna_predictor import (
    FEATURE_NAMES,
    LABELS,
    extract_features,
    labels_for_records,
    load_labeled_fasta,
    train_model,
)
from sequence_sources import SequenceHit, download_fasta, search_sequences


ROOT = Path(__file__).resolve().parent
DEMO_FASTA = ROOT / "data" / "lambda_labeled.fasta"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_RECORDS = 2500
MAX_SEQUENCE_LENGTH = 100_000
MAX_ARCHIVE_DOWNLOADS = 5

st.set_page_config(
    page_title="DNA Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f3f5f9; }
    [data-testid="stHeader"] { background: rgba(243, 245, 249, 0.9); }
    [data-testid="stSidebar"] { background: #17233b; }
    [data-testid="stSidebar"] * { color: #edf2fb; }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #17233b 0%, #294b82 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; margin-bottom: .25rem; }
    .hero p { color: #dce7f8; margin-bottom: 0; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dfe5ef;
        padding: 1rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_fasta_upload(content: bytes, filename: str) -> list[SeqRecord]:
    if not content:
        raise ValueError(f"{filename}: file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"{filename}: maximum upload size is 20 MiB")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{filename}: FASTA must be UTF-8 encoded") from error
    records = list(SeqIO.parse(io.StringIO(text), "fasta"))
    if not records:
        raise ValueError(f"{filename}: no FASTA records found")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"{filename}: maximum record count is {MAX_RECORDS}")
    for record in records:
        if len(record.seq) > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"FASTA record {record.id!r}: maximum sequence length is "
                f"{MAX_SEQUENCE_LENGTH:,} bases"
            )
    return records


def _feature_matrix(records: list[SeqRecord]) -> np.ndarray:
    rows = []
    for record in records:
        try:
            rows.append(extract_features(str(record.seq)))
        except ValueError as error:
            raise ValueError(f"FASTA record {record.id!r}: {error}") from error
    return np.vstack(rows)


def _fit(records: list[SeqRecord], test_fraction: float):
    labels = labels_for_records(records)
    model, x_test, y_test = train_model(
        records, labels, test_fraction=test_fraction, random_state=42
    )
    predictions = model.predict(x_test)
    return (
        model,
        float(accuracy_score(y_test, predictions)),
        classification_report(
            y_test,
            predictions,
            labels=list(LABELS),
            target_names=list(LABELS),
            zero_division=0,
        ),
        len(records),
        len(y_test),
    )


@st.cache_resource
def _get_demo_model():
    records, labels = load_labeled_fasta(DEMO_FASTA)
    model, x_test, y_test = train_model(records, labels, random_state=42)
    predictions = model.predict(x_test)
    return (
        model,
        float(accuracy_score(y_test, predictions)),
        classification_report(
            y_test,
            predictions,
            labels=list(LABELS),
            target_names=list(LABELS),
            zero_division=0,
        ),
        len(records),
        len(y_test),
    )


def _active_model():
    custom = st.session_state.get("custom_model")
    if custom is not None:
        return custom, "Your session model"
    demo, *_ = _get_demo_model()
    return demo, "Lambda phage demo model"


def _train_from_records(records: list[SeqRecord], test_fraction: float) -> None:
    model, accuracy, report, record_count, test_count = _fit(records, test_fraction)
    st.session_state["custom_model"] = model
    st.session_state["training_summary"] = {
        "accuracy": accuracy,
        "report": report,
        "record_count": record_count,
        "test_count": test_count,
    }


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>DNA Studio</h1>
          <p>Search public sequence archives, train a classifier, and inspect FASTA predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


_render_header()

with st.sidebar:
    st.markdown("## 🧬 DNA Studio")
    st.caption("A local-first sequence workspace")
    st.divider()
    st.markdown("**Model in this session**")
    if st.session_state.get("custom_model") is not None:
        st.success("Your trained model")
    else:
        st.info("Lambda demo baseline")
    st.caption(
        "Uploads, trained models, and prediction history are temporary and isolated "
        "to this browser session."
    )

predict_tab, train_tab, archive_tab, about_tab = st.tabs(
    ("Predict", "Train", "Archive search", "About & data")
)

with predict_tab:
    st.subheader("Predict FASTA records")
    model, model_name = _active_model()
    st.caption(f"Active model: **{model_name}**")
    uploaded = st.file_uploader(
        "Choose a FASTA file to classify",
        type=("fa", "fasta", "fna", "ffn"),
        key="prediction_upload",
    )
    if st.button("Run prediction", type="primary", disabled=uploaded is None):
        try:
            records = _read_fasta_upload(uploaded.getvalue(), uploaded.name)
            features = _feature_matrix(records)
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
            class_indices = {label: index for index, label in enumerate(model.classes_)}
            if "coding" not in class_indices:
                raise ValueError("Selected model does not provide a coding class")
            rows = [
                {
                    "sequence_id": record.id,
                    "predicted_label": str(prediction),
                    "coding_probability": float(probability[class_indices["coding"]]),
                    "gc_content": float(feature[0]),
                    "sequence_length": len(record.seq),
                }
                for record, prediction, probability, feature in zip(
                    records, predictions, probabilities, features
                )
            ]
            st.session_state["last_predictions"] = rows
            st.session_state["last_prediction_file"] = uploaded.name
        except (OSError, ValueError, TypeError, KeyError) as error:
            st.error(str(error))

    rows = st.session_state.get("last_predictions")
    if rows:
        st.markdown(f"#### Results — {st.session_state.get('last_prediction_file', '')}")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "Download predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name="dna_predictions.csv",
            mime="text/csv",
        )

with train_tab:
    st.subheader("Train your own model")
    st.write(
        "Upload a labeled FASTA with `label=coding` or `label=noncoding` in every "
        "header. Your model is held in this browser session; it is not saved or "
        "shared with other visitors."
    )
    st.download_button(
        "Download the documented demo training set",
        data=DEMO_FASTA.read_bytes(),
        file_name="lambda_labeled.fasta",
        mime="text/plain",
    )
    training_upload = st.file_uploader(
        "Labeled FASTA",
        type=("fa", "fasta", "fna", "ffn"),
        key="training_upload",
    )
    fraction = st.slider("Held-out test fraction", 0.1, 0.5, 0.2, 0.05)
    if st.button("Train from uploaded FASTA", type="primary", disabled=training_upload is None):
        try:
            records = _read_fasta_upload(training_upload.getvalue(), training_upload.name)
            labels_for_records(records)
            with st.spinner("Training and evaluating your model..."):
                _train_from_records(records, fraction)
        except (OSError, ValueError, TypeError) as error:
            st.error(str(error))
    if st.button("Train using the bundled reference set"):
        try:
            records, _labels = load_labeled_fasta(DEMO_FASTA)
            with st.spinner("Training and evaluating the bundled reference set..."):
                _train_from_records(records, fraction)
        except (OSError, ValueError, TypeError) as error:
            st.error(str(error))

    summary = st.session_state.get("training_summary")
    if summary:
        st.markdown("#### Held-out evaluation")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Accuracy", f"{summary['accuracy']:.3f}")
        metric_cols[1].metric("Training records", summary["record_count"])
        metric_cols[2].metric("Held-out records", summary["test_count"])
        st.code(summary["report"], language=None)
        st.warning(
            "A random split from a small or homologous dataset can overstate "
            "generalization. Use representative, non-overlapping data."
        )

with archive_tab:
    st.subheader("Search NCBI Nucleotide or ENA")
    st.caption(
        "Searches return public record metadata. Downloaded archive records are not "
        "automatically labeled coding/noncoding and cannot be used for supervised "
        "training until appropriately labeled."
    )
    source_label = st.selectbox("Archive", ("NCBI Nucleotide", "ENA"))
    query = st.text_input("Search term", placeholder="organism, gene, or sequence description")
    email = st.text_input("NCBI contact email (optional)")
    limit = st.select_slider("Maximum results", options=(10, 20, 50), value=10)
    if st.button("Search archive", type="primary"):
        source = "ncbi" if source_label == "NCBI Nucleotide" else "ena"
        try:
            with st.spinner("Searching the public archive..."):
                hits = search_sequences(source, query, limit, email)
            st.session_state["archive_hits"] = hits
            st.session_state["archive_source"] = source
            st.session_state.pop("archive_fasta", None)
            if not hits:
                st.info("No records found. Try a more specific or different search term.")
        except (OSError, ValueError, TypeError) as error:
            st.error(f"Archive search failed: {error}")

    hits = st.session_state.get("archive_hits", [])
    if hits:
        st.dataframe(
            [{"accession": hit.accession, "description": hit.title} for hit in hits],
            use_container_width=True,
            hide_index=True,
        )
        hit_by_label = {f"{hit.accession} — {hit.title[:100]}": hit for hit in hits}
        selected_labels = st.multiselect("Select records to download", tuple(hit_by_label))
        if len(selected_labels) > MAX_ARCHIVE_DOWNLOADS:
            st.warning(f"Select no more than {MAX_ARCHIVE_DOWNLOADS} records at a time.")
        if st.button(
            "Prepare FASTA download",
            disabled=not selected_labels or len(selected_labels) > MAX_ARCHIVE_DOWNLOADS,
        ):
            try:
                with st.spinner("Downloading selected records..."):
                    fasta_parts = []
                    total_bytes = 0
                    for label in selected_labels:
                        record = download_fasta(hit_by_label[label], email)
                        total_bytes += len(record.encode("utf-8"))
                        if total_bytes > MAX_UPLOAD_BYTES:
                            raise ValueError("Selected FASTA data exceeds the 20 MiB download limit")
                        fasta_parts.append(record)
                    fasta = "".join(fasta_parts)
                st.session_state["archive_fasta"] = fasta
                st.session_state["archive_filename"] = (
                    "ncbi_sequences.fasta"
                    if st.session_state.get("archive_source") == "ncbi"
                    else "ena_sequences.fasta"
                )
            except (OSError, ValueError, TypeError) as error:
                st.error(f"Download failed: {error}")
        if st.session_state.get("archive_fasta"):
            st.download_button(
                "Download selected FASTA",
                data=st.session_state["archive_fasta"],
                file_name=st.session_state["archive_filename"],
                mime="text/plain",
            )

with about_tab:
    st.subheader("About DNA Studio")
    st.caption("© 2026 Ishan Kundu. All rights reserved.")
    st.write(
        "DNA Studio is a small educational baseline using GC content and in-frame "
        "codon frequencies with standardized logistic regression. It is not a gene "
        "caller or a validated biological diagnostic."
    )
    st.markdown("#### Bundled reference data and model")
    st.write(
        "The demo training FASTA is a balanced sample of 18 coding sequences and 18 "
        "intergenic regions derived from the annotated bacteriophage lambda RefSeq "
        "record **NC_001416.1**. The baseline model is trained locally from that "
        "sample when the app starts. This single-genome dataset is illustrative and "
        "does not establish performance on other organisms."
    )
    st.markdown(
        "[NCBI RefSeq record NC_001416.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_001416.1) "
        "· [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/) "
        "· [ENA API documentation](https://ena-docs.readthedocs.io/en/latest/retrieval/programmatic-access.html)"
    )
    st.markdown("#### Privacy and resource limits")
    st.write(
        "FASTA uploads and custom models are processed in the current app session and "
        "are not written to shared storage. Uploads are limited to 20 MiB, 2,500 "
        "records, and 100,000 bases per record. Public hosting operators may still "
        "have infrastructure logs; review the host's privacy policy before uploading "
        "sensitive sequences."
    )
    st.info(
        "Each visitor trains a private, temporary session model. This public-ready "
        "version does not combine visitors' training data or keep models after a "
        "session ends."
    )

