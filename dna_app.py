"""Windows desktop interface for the DNA coding-region predictor."""

from __future__ import annotations

import csv
import pickle
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import joblib
import numpy as np
from Bio import SeqIO
from sklearn.metrics import accuracy_score

from dna_predictor import FEATURE_NAMES, LABELS, extract_features, load_labeled_fasta, train_model
from sequence_sources import SequenceHit, download_fasta, search_sequences


APP_DIR = Path.home() / ".dna_predictor"
DATABASE_PATH = APP_DIR / "dna_studio.sqlite3"
MODEL_PATH = APP_DIR / "dna_model.joblib"

COLORS = {
    "background": "#F3F5F9",
    "surface": "#FFFFFF",
    "sidebar": "#17233B",
    "sidebar_active": "#293B5F",
    "text": "#1B2740",
    "muted": "#66738A",
    "accent": "#3768D5",
    "accent_dark": "#2853B1",
    "border": "#DFE5EF",
    "success": "#16815C",
    "error": "#B54747",
}


class DnaStudio(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        APP_DIR.mkdir(parents=True, exist_ok=True)
        self.title("DNA Studio | Coding Region Predictor")
        self.geometry("1180x780")
        self.minsize(980, 660)
        self.configure(background=COLORS["background"])

        self.status_var = tk.StringVar(value="Ready")
        self.model_var = tk.StringVar()
        self.training_file_var = tk.StringVar(value="No labeled FASTA selected")
        self.prediction_file_var = tk.StringVar(value="No FASTA selected")
        self.search_results: list[SequenceHit] = []
        self.model_path = MODEL_PATH

        self._init_database()
        self._init_styles()
        self._build_shell()
        self._build_pages()
        self._refresh_model_status()
        self._refresh_results()
        self._show_page("Dashboard")

    def _init_database(self) -> None:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    sequence_id TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    coding_probability REAL NOT NULL,
                    gc_content REAL NOT NULL,
                    sequence_length INTEGER NOT NULL
                )
                """
            )

    def _init_styles(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=COLORS["background"])
        style.configure("Page.TFrame", background=COLORS["background"])
        style.configure("Card.TFrame", background=COLORS["surface"])
        style.configure(
            "TLabel",
            background=COLORS["background"],
            foreground=COLORS["text"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Title.TLabel",
            background=COLORS["background"],
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 22),
        )
        style.configure(
            "Subtitle.TLabel",
            background=COLORS["background"],
            foreground=COLORS["muted"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=COLORS["background"],
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 13),
        )
        style.configure(
            "CardTitle.TLabel",
            background=COLORS["surface"],
            foreground=COLORS["muted"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "CardValue.TLabel",
            background=COLORS["surface"],
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 17),
        )
        style.configure(
            "Accent.TButton",
            background=COLORS["accent"],
            foreground="#FFFFFF",
            padding=(14, 8),
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "Accent.TButton",
            background=[("active", COLORS["accent_dark"]), ("disabled", "#A9B6CF")],
            foreground=[("disabled", "#FFFFFF")],
        )
        style.configure("Treeview", rowheight=29, font=("Segoe UI", 9))
        style.configure(
            "Treeview.Heading",
            font=("Segoe UI Semibold", 9),
            background="#E9EDF5",
            foreground=COLORS["text"],
        )

    def _build_shell(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=COLORS["surface"], height=76)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        tk.Label(
            header,
            text="DNA Studio",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 18),
        ).pack(side="left", padx=(26, 8), pady=18)
        tk.Label(
            header,
            text="Coding region predictor",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
        ).pack(side="left", pady=22)
        self.model_status_label = tk.Label(
            header,
            textvariable=self.model_var,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI Semibold", 9),
        )
        self.model_status_label.pack(side="right", padx=26)
        tk.Frame(self, bg=COLORS["border"], height=1).grid(
            row=0, column=0, columnspan=2, sticky="sew"
        )

        self.sidebar = tk.Frame(self, bg=COLORS["sidebar"], width=220)
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        tk.Label(
            self.sidebar,
            text="WORKSPACE",
            bg=COLORS["sidebar"],
            fg="#9EACC5",
            font=("Segoe UI Semibold", 8),
        ).pack(anchor="w", padx=22, pady=(27, 12))

        self.nav_buttons: dict[str, tk.Button] = {}
        for page in ("Dashboard", "Archive Search", "Train Model", "Predict", "Results"):
            button = tk.Button(
                self.sidebar,
                text=page,
                anchor="w",
                padx=20,
                pady=11,
                borderwidth=0,
                relief="flat",
                bg=COLORS["sidebar"],
                fg="#DFE7F5",
                activebackground=COLORS["sidebar_active"],
                activeforeground="#FFFFFF",
                font=("Segoe UI", 10),
                command=lambda name=page: self._show_page(name),
            )
            button.pack(fill="x", padx=10, pady=2)
            self.nav_buttons[page] = button

        tk.Label(
            self.sidebar,
            text="LOCAL WORKSPACE",
            bg=COLORS["sidebar"],
            fg="#9EACC5",
            font=("Segoe UI Semibold", 8),
        ).pack(anchor="w", padx=22, pady=(32, 6))
        tk.Label(
            self.sidebar,
            text="Model and prediction history\nstay on this computer.",
            justify="left",
            bg=COLORS["sidebar"],
            fg="#DFE7F5",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=22, pady=4)
        tk.Label(
            self.sidebar,
            text="(c) 2026 Ishan Kundu\nAll rights reserved.",
            justify="left",
            bg=COLORS["sidebar"],
            fg="#9EACC5",
            font=("Segoe UI", 8),
        ).pack(anchor="w", padx=22, pady=(16, 0))

        self.content = tk.Frame(self, bg=COLORS["background"])
        self.content.grid(row=1, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        footer = tk.Frame(self, bg=COLORS["surface"], height=32)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        footer.grid_propagate(False)
        self.status_label = tk.Label(
            footer,
            textvariable=self.status_var,
            anchor="w",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        )
        self.status_label.pack(fill="both", padx=14)

    def _build_pages(self) -> None:
        self.pages: dict[str, ttk.Frame] = {}
        for name in ("Dashboard", "Archive Search", "Train Model", "Predict", "Results"):
            page = ttk.Frame(self.content, style="Page.TFrame", padding=26)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[name] = page
        self._build_dashboard()
        self._build_archive_search()
        self._build_train_page()
        self._build_predict_page()
        self._build_results_page()

    def _page_heading(self, page: ttk.Frame, title: str, subtitle: str) -> None:
        ttk.Label(page, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(page, text=subtitle, style="Subtitle.TLabel").pack(
            anchor="w", pady=(5, 20)
        )

    @staticmethod
    def _card(parent: tk.Misc, title: str, value: str) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        ttk.Label(card, text=title.upper(), style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text=value, style="CardValue.TLabel").pack(
            anchor="w", pady=(7, 0)
        )
        return card

    def _build_dashboard(self) -> None:
        page = self.pages["Dashboard"]
        self._page_heading(
            page,
            "Welcome to DNA Studio",
            "A local workspace for sequence search, training, and prediction.",
        )

        cards = ttk.Frame(page, style="Page.TFrame")
        cards.pack(fill="x", pady=(0, 20))
        for column in range(3):
            cards.columnconfigure(column, weight=1, uniform="cards")
        self.dashboard_model_card = self._card(cards, "Model", "Not trained")
        self.dashboard_model_card.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.dashboard_result_card = self._card(cards, "Saved predictions", "0")
        self.dashboard_result_card.grid(row=0, column=1, sticky="ew", padx=8)
        self._card(cards, "Sequence archives", "NCBI + ENA").grid(
            row=0, column=2, sticky="ew", padx=(8, 0)
        )

        intro = ttk.Frame(page, style="Card.TFrame", padding=22)
        intro.pack(fill="x")
        tk.Label(
            intro,
            text="A simple path from FASTA to results",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 13),
        ).pack(anchor="w")
        for text in (
            "1. Search public NCBI Nucleotide or ENA records, or import your own FASTA.",
            "2. Train with labeled FASTA records (headers must include label=coding or label=noncoding).",
            "3. Predict coding probability, review saved results, and export a CSV.",
        ):
            tk.Label(
                intro,
                text=text,
                bg=COLORS["surface"],
                fg=COLORS["muted"],
                font=("Segoe UI", 10),
                anchor="w",
            ).pack(anchor="w", pady=(12, 0))

        tk.Label(
            page,
            text=(
                "Model scope: this baseline uses GC content and frame-0 codon frequencies; "
                "it is not a gene caller. Archive results are public records, not curated "
                "coding/noncoding labels. WHO is not a general public nucleotide FASTA archive."
            ),
            wraplength=850,
            justify="left",
            bg=COLORS["background"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(18, 0))

    def _build_archive_search(self) -> None:
        page = self.pages["Archive Search"]
        self._page_heading(
            page,
            "Find sequence records",
            "Search archive metadata, select records, and save their sequences as a FASTA file.",
        )
        form = ttk.Frame(page, style="Card.TFrame", padding=18)
        form.pack(fill="x", pady=(0, 16))
        ttk.Label(form, text="Archive").grid(row=0, column=0, sticky="w")
        self.archive_var = tk.StringVar(value="NCBI Nucleotide")
        ttk.Combobox(
            form,
            textvariable=self.archive_var,
            values=("NCBI Nucleotide", "ENA"),
            state="readonly",
            width=22,
        ).grid(row=1, column=0, sticky="w", pady=(5, 12), padx=(0, 16))

        ttk.Label(form, text="Search term").grid(row=0, column=1, sticky="w")
        self.query_var = tk.StringVar()
        query_entry = ttk.Entry(form, textvariable=self.query_var, width=38)
        query_entry.grid(row=1, column=1, sticky="ew", pady=(5, 12), padx=(0, 16))
        query_entry.bind("<Return>", lambda _event: self._search_archive())

        ttk.Label(form, text="Max records").grid(row=0, column=2, sticky="w")
        self.limit_var = tk.StringVar(value="20")
        ttk.Combobox(
            form,
            textvariable=self.limit_var,
            values=("10", "20", "50"),
            state="readonly",
            width=9,
        ).grid(row=1, column=2, sticky="w", pady=(5, 12), padx=(0, 16))
        ttk.Label(form, text="NCBI contact email (optional)").grid(
            row=2, column=0, columnspan=2, sticky="w"
        )
        self.email_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.email_var, width=38).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(5, 12)
        )
        self.search_button = ttk.Button(
            form, text="Search archive", style="Accent.TButton", command=self._search_archive
        )
        self.search_button.grid(row=1, column=3, sticky="e")
        form.columnconfigure(1, weight=1)

        result_header = ttk.Frame(page, style="Page.TFrame")
        result_header.pack(fill="x", pady=(0, 8))
        ttk.Label(result_header, text="Search results", style="Section.TLabel").pack(side="left")
        self.download_button = ttk.Button(
            result_header,
            text="Download selected FASTA",
            command=self._download_selected,
        )
        self.download_button.pack(side="right")

        table_frame = ttk.Frame(page)
        table_frame.pack(fill="both", expand=True)
        self.search_tree = ttk.Treeview(
            table_frame,
            columns=("accession", "title"),
            show="headings",
            selectmode="extended",
        )
        self.search_tree.heading("accession", text="ACCESSION")
        self.search_tree.heading("title", text="RECORD DESCRIPTION")
        self.search_tree.column("accession", width=170, stretch=False)
        self.search_tree.column("title", width=640)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.search_tree.yview)
        self.search_tree.configure(yscrollcommand=scrollbar.set)
        self.search_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        ttk.Label(
            page,
            text="Search uses public sequence metadata. Large sequences are downloaded only for selected records.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(10, 0))

    def _build_train_page(self) -> None:
        page = self.pages["Train Model"]
        self._page_heading(
            page,
            "Train a model",
            "Choose a labeled FASTA dataset to fit and evaluate a new local classifier.",
        )
        panel = ttk.Frame(page, style="Card.TFrame", padding=20)
        panel.pack(fill="x")
        tk.Label(
            panel,
            text="Training dataset",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 12),
        ).pack(anchor="w")
        tk.Label(
            panel,
            textvariable=self.training_file_var,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(8, 14))
        actions = ttk.Frame(panel, style="Card.TFrame")
        actions.pack(anchor="w")
        ttk.Button(actions, text="Choose labeled FASTA", command=self._choose_training_file).pack(
            side="left"
        )
        self.train_button = ttk.Button(
            actions,
            text="Train and evaluate",
            style="Accent.TButton",
            command=self._train_model,
        )
        self.train_button.pack(side="left", padx=(10, 0))
        self.training_report = tk.Text(
            page,
            height=14,
            wrap="word",
            state="disabled",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            relief="flat",
            padx=14,
            pady=12,
            font=("Consolas", 10),
        )
        self.training_report.pack(fill="both", expand=True, pady=(18, 0))
        self._set_text(
            self.training_report,
            "Training report will appear here. Provide at least two records per class; "
            "each FASTA header must contain label=coding or label=noncoding.",
        )

    def _build_predict_page(self) -> None:
        page = self.pages["Predict"]
        self._page_heading(
            page,
            "Predict sequences",
            "Classify each FASTA record with the currently saved model.",
        )
        panel = ttk.Frame(page, style="Card.TFrame", padding=20)
        panel.pack(fill="x")
        tk.Label(
            panel,
            text="Prediction input",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 12),
        ).pack(anchor="w")
        tk.Label(
            panel,
            textvariable=self.prediction_file_var,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(8, 14))
        actions = ttk.Frame(panel, style="Card.TFrame")
        actions.pack(anchor="w")
        ttk.Button(actions, text="Choose FASTA", command=self._choose_prediction_file).pack(
            side="left"
        )
        self.predict_button = ttk.Button(
            actions,
            text="Run prediction",
            style="Accent.TButton",
            command=self._predict,
        )
        self.predict_button.pack(side="left", padx=(10, 0))
        ttk.Button(actions, text="Choose model file", command=self._choose_model_file).pack(
            side="left", padx=(10, 0)
        )
        self.prediction_summary = tk.Label(
            page,
            text="Results are saved locally and can be exported from the Results page.",
            bg=COLORS["background"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.prediction_summary.pack(fill="x", pady=(18, 0))

    def _build_results_page(self) -> None:
        page = self.pages["Results"]
        self._page_heading(
            page,
            "Prediction history",
            "Review previous predictions stored on this computer, or export them as CSV.",
        )
        actions = ttk.Frame(page, style="Page.TFrame")
        actions.pack(fill="x", pady=(0, 10))
        ttk.Button(actions, text="Refresh", command=self._refresh_results).pack(side="left")
        ttk.Button(
            actions,
            text="Export all to CSV",
            style="Accent.TButton",
            command=self._export_results,
        ).pack(side="right")

        table_frame = ttk.Frame(page)
        table_frame.pack(fill="both", expand=True)
        columns = (
            "created_at",
            "sequence_id",
            "predicted_label",
            "coding_probability",
            "gc_content",
            "sequence_length",
            "source_file",
        )
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        headings = {
            "created_at": "DATE",
            "sequence_id": "SEQUENCE",
            "predicted_label": "PREDICTION",
            "coding_probability": "CODING PROB.",
            "gc_content": "GC CONTENT",
            "sequence_length": "LENGTH",
            "source_file": "SOURCE FILE",
        }
        widths = {
            "created_at": 145,
            "sequence_id": 130,
            "predicted_label": 110,
            "coding_probability": 110,
            "gc_content": 100,
            "sequence_length": 80,
            "source_file": 260,
        }
        for column in columns:
            self.results_tree.heading(column, text=headings[column])
            self.results_tree.column(column, width=widths[column], anchor="w")
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

    def _show_page(self, name: str) -> None:
        self.pages[name].tkraise()
        for page_name, button in self.nav_buttons.items():
            button.configure(
                bg=COLORS["sidebar_active"] if page_name == name else COLORS["sidebar"]
            )
        if name == "Results":
            self._refresh_results()

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status_var.set(message)
        self.status_label.configure(fg=COLORS["error"] if error else COLORS["muted"])

    def _run_background(self, button: ttk.Button, status: str, work, on_success) -> None:
        button.state(["disabled"])
        self._set_status(status)

        def runner() -> None:
            try:
                result = work()
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                EOFError,
                sqlite3.Error,
                pickle.UnpicklingError,
                ImportError,
                AttributeError,
            ) as error:
                self.after(0, lambda error=error: self._task_failed(button, error))
            else:
                self.after(
                    0, lambda result=result: self._task_succeeded(button, on_success, result)
                )

        threading.Thread(target=runner, daemon=True).start()

    def _task_failed(self, button: ttk.Button, error: Exception) -> None:
        button.state(["!disabled"])
        self._set_status("Operation failed", error=True)
        messagebox.showerror("Could not complete operation", str(error), parent=self)

    def _task_succeeded(self, button: ttk.Button, on_success, result) -> None:
        button.state(["!disabled"])
        on_success(result)

    def _choose_training_file(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self,
            title="Choose labeled FASTA",
            filetypes=(("FASTA files", "*.fa *.fasta *.fna *.ffn *.fas"), ("All files", "*.*")),
        )
        if selected:
            self.training_file_var.set(selected)

    def _train_model(self) -> None:
        selected = self.training_file_var.get()
        if selected == "No labeled FASTA selected" or not Path(selected).is_file():
            messagebox.showinfo(
                "Training data needed", "Choose a labeled FASTA file first.", parent=self
            )
            return

        def work():
            records, labels = load_labeled_fasta(Path(selected))
            model, x_test, y_test = train_model(records, labels)
            predictions = model.predict(x_test)
            accuracy = float(accuracy_score(y_test, predictions))
            joblib.dump(
                {"model": model, "feature_names": FEATURE_NAMES, "labels": LABELS},
                MODEL_PATH,
            )
            return accuracy, len(records), len(y_test)

        def completed(result) -> None:
            accuracy, record_count, test_count = result
            self._set_text(
                self.training_report,
                f"Training complete\n\n"
                f"Records used: {record_count}\n"
                f"Held-out records: {test_count}\n"
                f"Held-out accuracy: {accuracy:.3f}\n\n"
                f"Model saved to:\n{MODEL_PATH}\n\n"
                "Evaluation metrics depend on the representativeness and independence "
                "of the supplied sequences.",
            )
            self.model_path = MODEL_PATH
            self._refresh_model_status()
            self._set_status("Model trained and saved")

        self._run_background(self.train_button, "Training model...", work, completed)

    def _choose_prediction_file(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self,
            title="Choose FASTA to predict",
            filetypes=(("FASTA files", "*.fa *.fasta *.fna *.ffn *.fas"), ("All files", "*.*")),
        )
        if selected:
            self.prediction_file_var.set(selected)

    def _choose_model_file(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self,
            title="Choose a trained model",
            filetypes=(("Joblib model", "*.joblib"), ("All files", "*.*")),
        )
        if selected:
            self.model_path = Path(selected)
            self._refresh_model_status()

    def _predict(self) -> None:
        selected = self.prediction_file_var.get()
        if selected == "No FASTA selected" or not Path(selected).is_file():
            messagebox.showinfo("Input needed", "Choose a FASTA file first.", parent=self)
            return
        model_path = self.model_path
        if not model_path.is_file():
            messagebox.showinfo(
                "Model needed",
                "Train a model or choose an existing .joblib model file first.",
                parent=self,
            )
            return

        def work():
            bundle = joblib.load(model_path)
            if (
                not isinstance(bundle, dict)
                or bundle.get("feature_names") != FEATURE_NAMES
                or not {"model", "labels"}.issubset(bundle)
            ):
                raise ValueError("Model file is incompatible with this version of DNA Studio")
            with Path(selected).open(encoding="utf-8-sig") as fasta_file:
                records = list(SeqIO.parse(fasta_file, "fasta"))
            if not records:
                raise ValueError(f"No FASTA records found in {selected}")

            feature_rows = []
            for record in records:
                try:
                    feature_rows.append(extract_features(str(record.seq)))
                except ValueError as error:
                    raise ValueError(f"FASTA record {record.id!r}: {error}") from error
            features = np.vstack(feature_rows)
            model = bundle["model"]
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
            class_indices = {label: index for index, label in enumerate(model.classes_)}
            if "coding" not in class_indices:
                raise ValueError("The selected model does not provide a coding class")

            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            rows = []
            for record, prediction, probability, feature in zip(
                records, predictions, probabilities, features
            ):
                rows.append(
                    (
                        timestamp,
                        str(Path(selected)),
                        record.id,
                        str(prediction),
                        float(probability[class_indices["coding"]]),
                        float(feature[0]),
                        len(record.seq),
                    )
                )
            with sqlite3.connect(DATABASE_PATH) as connection:
                connection.executemany(
                    """
                    INSERT INTO predictions (
                        created_at, source_file, sequence_id, predicted_label,
                        coding_probability, gc_content, sequence_length
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            return rows

        def completed(rows) -> None:
            self._refresh_results()
            self.prediction_summary.configure(
                text=f"Prediction complete: {len(rows)} record(s) added to local history."
            )
            self._set_status(f"Predicted {len(rows)} sequence(s)")
            self._show_page("Results")

        self._run_background(self.predict_button, "Predicting sequences...", work, completed)

    def _search_archive(self) -> None:
        source = "ncbi" if self.archive_var.get() == "NCBI Nucleotide" else "ena"
        query = self.query_var.get()
        email = self.email_var.get()
        try:
            limit = int(self.limit_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid result limit", "Choose 10, 20, or 50 records.", parent=self
            )
            return

        def work():
            return search_sequences(source, query, limit, email)

        def completed(hits: list[SequenceHit]) -> None:
            self.search_results = hits
            self.search_tree.delete(*self.search_tree.get_children())
            for index, hit in enumerate(hits):
                self.search_tree.insert(
                    "",
                    "end",
                    iid=str(index),
                    values=(hit.accession, hit.title),
                )
            self._set_status(f"Found {len(hits)} record(s)")
            if not hits:
                messagebox.showinfo("No records found", "Try another search term.", parent=self)

        self._run_background(self.search_button, "Searching sequence archive...", work, completed)

    def _download_selected(self) -> None:
        selected_indices = self.search_tree.selection()
        if not selected_indices:
            messagebox.showinfo(
                "Select records", "Select one or more search results first.", parent=self
            )
            return
        if len(selected_indices) > 5:
            messagebox.showinfo(
                "Download limit",
                "Select no more than five records in one download.",
                parent=self,
            )
            return
        target = filedialog.asksaveasfilename(
            parent=self,
            title="Save selected sequences",
            defaultextension=".fasta",
            filetypes=(("FASTA files", "*.fasta"), ("All files", "*.*")),
            initialfile="archive_sequences.fasta",
        )
        if not target:
            return
        hits = [self.search_results[int(index)] for index in selected_indices]
        email = self.email_var.get()

        def work():
            content = []
            total_bytes = 0
            for hit in hits:
                record = download_fasta(hit, email)
                total_bytes += len(record.encode("utf-8"))
                if total_bytes > 20 * 1024 * 1024:
                    raise ValueError("Selected FASTA data exceeds the 20 MiB download limit")
                content.append(record)
            Path(target).write_text("".join(content), encoding="utf-8")
            return len(content)

        def completed(count: int) -> None:
            self._set_status(f"Saved {count} sequence(s) to {target}")
            messagebox.showinfo(
                "FASTA saved", f"Downloaded {count} sequence(s) to:\n{target}", parent=self
            )

        self._run_background(
            self.download_button, "Downloading selected sequences...", work, completed
        )

    def _refresh_results(self) -> None:
        if not hasattr(self, "results_tree"):
            return
        with sqlite3.connect(DATABASE_PATH) as connection:
            rows = connection.execute(
                """
                SELECT created_at, sequence_id, predicted_label, coding_probability,
                       gc_content, sequence_length, source_file
                FROM predictions ORDER BY id DESC
                """
            ).fetchall()
            count = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        self.results_tree.delete(*self.results_tree.get_children())
        for row in rows:
            self.results_tree.insert(
                "",
                "end",
                values=(
                    row[0],
                    row[1],
                    row[2],
                    f"{row[3]:.6f}",
                    f"{row[4]:.4f}",
                    row[5],
                    row[6],
                ),
            )
        self._update_card(self.dashboard_result_card, "Saved predictions", str(count))

    def _export_results(self) -> None:
        target = filedialog.asksaveasfilename(
            parent=self,
            title="Export prediction history",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            initialfile="dna_predictions.csv",
        )
        if not target:
            return
        with sqlite3.connect(DATABASE_PATH) as connection:
            rows = connection.execute(
                """
                SELECT created_at, source_file, sequence_id, predicted_label,
                       coding_probability, gc_content, sequence_length
                FROM predictions ORDER BY id
                """
            ).fetchall()
        with Path(target).open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(
                (
                    "created_at",
                    "source_file",
                    "sequence_id",
                    "predicted_label",
                    "coding_probability",
                    "gc_content",
                    "sequence_length",
                )
            )
            writer.writerows(rows)
        self._set_status(f"Exported {len(rows)} row(s) to {target}")
        messagebox.showinfo("CSV exported", f"Saved results to:\n{target}", parent=self)

    def _refresh_model_status(self) -> None:
        if self.model_path.is_file():
            self.model_var.set(f"Model ready  |  {self.model_path.name}")
            self.model_status_label.configure(fg=COLORS["success"])
            self._update_card(self.dashboard_model_card, "Model", "Ready")
        else:
            self.model_var.set("No trained model")
            self.model_status_label.configure(fg=COLORS["muted"])
            self._update_card(self.dashboard_model_card, "Model", "Not trained")

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    @staticmethod
    def _update_card(card: ttk.Frame, title: str, value: str) -> None:
        labels = [child for child in card.winfo_children() if isinstance(child, ttk.Label)]
        if len(labels) >= 2:
            labels[0].configure(text=title.upper())
            labels[1].configure(text=value)


if __name__ == "__main__":
    DnaStudio().mainloop()
