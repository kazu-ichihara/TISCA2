# TISCA2: Translation Initiation Site detection by translation Complex Analysis 2

TISCA2 is a Python pipeline for identifying translation initiation sites (TIS) from eIF5B Sel-TCP-seq data mapped to the transcriptome.

## Pipeline Overview

```
Step 1   GTF + FASTA  →  translon list
Step 2   BAM          →  metagene heatmap
Step 3   BAM + Step 2 →  cosine similarity scores
Step 4   Step 3       →  filtered & classified TIS candidates
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/TISCA2.git
cd TISCA2
pip install -e .
```

**Requirements:** Python >= 3.10

Dependencies are installed automatically:
`numpy`, `pandas`, `biopython`, `pysam`, `tqdm`, `scikit-learn`, `scipy`, `matplotlib`

## Usage

### Step 1: Generate translon list

```bash
tisca2-step1 \
  --gtf   ./ref/annotation.gtf \
  --fasta ./ref/transcript.fasta \
  --output ./transcript/ \
  --name  sample \
  [--codons ATG CTG GTG ...]
```

**Outputs:**  
- `transcript/annotated_translon.csv` — Annotated CDS coordinates (transcript space)  
- `transcript/symbol.csv` — transcript ID → gene symbol mapping  
- `transcript/{name}_translon.csv` — All ORF (translon) records  
- `transcript/{name}_translon_dict.pickle` — Translon start position dictionary  
- `transcript/{name}_translon_position.csv` — Genomic coordinate map

### Step 2: Build metagene heatmap

```bash
tisca2-step2 \
  --bam    mapped.bam \
  --ref    ./transcript/annotated_translon.csv \
  --symbol ./transcript/symbol.csv \
  --output ./transcript/ \
  --name   sample \
  [--upstream 50] [--downstream 30] \
  [--rlen-min 20] [--rlen-max 80] [--rlen-cap 90] \
  [--min-reads 10]
```

**Outputs:**  
- `transcript/{name}_heatplot_start.npy` — Metagene coverage matrix  
- `transcript/{name}_heatplot_start_diff.npy` — Differential metagene matrix

### Step 3: Cosine similarity scoring

```bash
tisca2-step3 \
  --bam      mapped.bam \
  --metagene ./transcript/{name}_heatplot_start_diff.npy \
  --translon ./transcript/{name}_translon_dict.pickle \
  --symbol   ./transcript/symbol.csv \
  --output   ./transcript/ \
  --name     sample \
  [--half-window 30] [--meta-trim 20] \
  [--rlen-min 20] [--rlen-max 80] [--rlen-cap 150] \
  [--min-reads 10]
```

**Outputs:**  
- `transcript/{name}_cosine.csv` — Cosine similarity scores per translon

### Step 4: Filtering and classification

```bash
# Threshold derived automatically from precision-recall curve
tisca2-step4 \
  --fasta    ./ref/transcript.fasta \
  --cosine   ./transcript/{name}_cosine.csv \
  --position ./transcript/{name}_translon_position.csv \
  --symbol   ./transcript/symbol.csv \
  --ref      ./transcript/annotated_translon.csv \
  --output   ./transcript/ \
  --name     sample

# Or specify a threshold manually
tisca2-step4 ... --threshold 0.5
```

**Outputs:**  
- `transcript/{name}_cosine_eff.csv` — Filtered cosine scores with annotation  
- `transcript/{name}_PR_curve.png` — Precision-recall curve (auto-threshold only)  
- `transcript/{name}_PR_threshold.png` — PR/F1 curves with threshold line (auto-threshold only)  
- `transcript/{name}_results.csv` — Final classified TIS candidates

## Translon Type Classification

| Type | Description |
|---|---|
| `canonical` | Matches annotated CDS start and stop |
| `uORF` | Upstream, out-of-frame, stops before CDS |
| `overlap.uORF` | Upstream, overlaps CDS start |
| `extension` | In-frame, starts upstream of CDS |
| `truncation` | In-frame, starts downstream of CDS |
| `internal` | Entirely within CDS |
| `external` | Starts within CDS, stops downstream |
| `polycistronic` | Spans beyond CDS stop |
| `noncoding` | On a non-coding transcript |
| `novel` | On an unannotated transcript (prefix: `MSTRG`) |

## Running Tests

```bash
pip install pytest
pytest tests/
```

## Citation

If you use TISCA2 in your research, please cite:

> https://www.biorxiv.org/content/10.64898/2026.11.03.763266v1
