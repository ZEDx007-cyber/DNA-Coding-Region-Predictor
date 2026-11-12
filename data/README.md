# Demo reference data

`lambda_labeled.fasta` is a small, balanced training sample derived from the
annotated bacteriophage lambda record **NC_001416.1** (48,502 bases), retrieved
from NCBI RefSeq / GenBank.

The file contains 18 CDS features labeled `coding` and 18 intergenic sequence
gaps labeled `noncoding`. Coding sequences are extracted in their annotated
strand orientation. Intergenic gaps are selected between annotated CDS, tRNA,
rRNA, ncRNA, and tmRNA features. Only examples 90–3,000 bases long are retained;
the sample is evenly subsampled across the record. These labels describe the
selected annotated feature/gap classes and should not be read as a complete
biological ground truth for transcription.

The bundled model is trained from this FASTA in memory when the web app starts.
It is a demonstration baseline for one viral reference, not a validated model
for other organisms. Its held-out score can be optimistic because the small
random split does not provide an independent genome-level test.

The sequence data is sourced from a third-party NCBI record. This project does
not claim ownership of the underlying sequence or upstream annotations; users
should follow the source database's current terms and attribution guidance.
