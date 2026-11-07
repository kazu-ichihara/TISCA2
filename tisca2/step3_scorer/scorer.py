"""
TISCA2 - Step 3: 2D Sliding Window and Cosine Similarity Scoring

Build a per-transcript read-length-stratified coverage matrix (2D sliding
window) from a transcriptome-mapped BAM file, then compute the cosine
similarity between each translon window and the reference differential
metagene profile from Step 2.

Usage:
    python -m tisca2.step3_scorer.scorer \
        --bam        mapped.bam \
        --metagene   ./transcript/{name}_heatplot_start_diff.npy \
        --translon   ./transcript/{name}_translon_dict.pickle \
        --symbol     ./transcript/symbol.csv \
        --output     ./transcript/ \
        --name       sample \
        [--half-window 30] [--meta-trim 20] \
        [--rlen-min 20] [--rlen-max 80] [--rlen-cap 150] \
        [--min-reads 10]

Outputs
-------
{name}_cosine.csv   columns: transcript ID, start, cosine similarity, length coverage
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.
    Returns 0.0 if either vector has zero norm.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ---------------------------------------------------------------------------
# Per-transcript coverage matrix
# ---------------------------------------------------------------------------

def build_coverage(
    bamfile: pysam.AlignmentFile,
    name: str,
    length: int,
    rlen_cap: int = 150,
) -> tuple[np.ndarray, int]:
    """
    Accumulate a read-length x position coverage matrix for one transcript.

    Parameters
    ----------
    bamfile : pysam.AlignmentFile
    name : str
        Transcript ID (reference name in BAM).
    length : int
        Transcript length in nucleotides.
    rlen_cap : int
        Read lengths >= rlen_cap are discarded.

    Returns
    -------
    m : np.ndarray, shape (rlen_cap, length)
    n_reads : int
        Number of reads that passed the rlen_cap filter.
    """
    m       = np.zeros((rlen_cap, length))
    n_reads = 0
    for read in bamfile.fetch(name):
        l = read.rlen
        if l >= rlen_cap:
            continue
        n_reads += 1
        for p in read.get_reference_positions():
            if p < length:
                m[l, p] += 1
    return m, n_reads


# ---------------------------------------------------------------------------
# Cosine scoring for all translons of one transcript
# ---------------------------------------------------------------------------

def score_translons(
    dist: np.ndarray,
    ref_meta: np.ndarray,
    starts: np.ndarray,
    half_window: int = 30,
) -> list:
    """
    Compute cosine similarity for each translon start position.

    For each start, a window of [start-half_window, start+half_window) is
    extracted from the coverage matrix (dist) and its column-wise diff is
    compared against the reference differential metagene (ref_meta).

    Boundary handling
    -----------------
    When the window is clipped at the left edge (left == 0) or right edge
    (right == length), the reference metagene slice is adjusted to match
    the clipped window size so the shapes align for cosine similarity.

    Parameters
    ----------
    dist : np.ndarray, shape (n_rlen, length)
        Read-length-stratified coverage for the transcript (rlen_min:rlen_max slice).
    ref_meta : np.ndarray, shape (n_rlen, n_ref_cols)
        Reference differential metagene from Step 2 (after meta-trim).
    starts : np.ndarray
        Translon start positions (transcript coordinates).
    half_window : int
        Half-width of the sliding window.

    Returns
    -------
    records : list of [start, cosine_similarity, length_coverage]
    """
    length  = dist.shape[1]
    records = []

    for start in starts:
        left  = max(0, start - half_window)
        right = min(length, start + half_window)

        d     = dist[:, left:right]
        z     = np.diff(d, axis=1)
        l_cov = int(np.count_nonzero(d.sum(axis=1)))

        if left == 0:
            ref_slice = ref_meta[:, ref_meta.shape[1] - right:]
        elif right == length:
            ref_cols  = right - left - 1
            ref_slice = ref_meta[:, :ref_cols]
        else:
            ref_slice = ref_meta

        ref_z    = np.diff(ref_slice, axis=1)
        min_cols = min(z.shape[1], ref_z.shape[1])
        if min_cols == 0:
            continue

        s = cos_sim(np.ravel(z[:, :min_cols]), np.ravel(ref_z[:, :min_cols]))
        records.append([start, s, l_cov])

    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    bam_path: str,
    metagene_path: str,
    translon_path: str,
    symbol_path: str,
    output_dir: str,
    fname: str,
    half_window: int,
    meta_trim: int,
    rlen_min: int,
    rlen_max: int,
    rlen_cap: int,
    min_reads: int,
) -> None:
    """Run Step 3+4: build coverage matrices and compute cosine similarities."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'[Step 3] Loading differential metagene: {metagene_path}')
    ref_full = np.load(metagene_path)
    ref_meta = ref_full[:, meta_trim:]

    print(f'[Step 3] Loading translon dict: {translon_path}')
    with open(translon_path, 'rb') as fh:
        start_dict = pickle.load(fh)

    print(f'[Step 3] Loading symbol table: {symbol_path}')
    symbol   = pd.read_csv(symbol_path, index_col=0)
    len_dict = symbol['length'].to_dict()

    names   = list(start_dict.keys())
    results = []

    bamfile = pysam.AlignmentFile(bam_path, 'rb')
    print(f'[Step 3] Scoring {len(names):,} transcripts ...')

    for name in tqdm(names):
        if name not in len_dict:
            continue
        length = len_dict[name]

        m, n_reads = build_coverage(bamfile, name, length, rlen_cap)
        if n_reads < min_reads:
            continue

        dist    = m[rlen_min:rlen_max]
        starts  = start_dict[name]
        records = score_translons(dist, ref_meta, starts, half_window)
        for start, s, l_cov in records:
            results.append([name, start, s, l_cov])

    bamfile.close()

    df      = pd.DataFrame(results, columns=['transcript ID', 'start', 'cosine similarity', 'length coverage'])
    out_csv = out / f'{fname}_cosine.csv'
    df.to_csv(out_csv, index=False)
    print(f'[Step 3] Saved cosine scores: {out_csv}  ({len(df):,} records)')
    print('[Step 3] Done.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="tisca2-step3",
        description='TISCA2 Step 3: 2D sliding window and cosine similarity scoring.'
    )
    parser.add_argument('--bam',      required=True, help='Path to transcriptome-mapped BAM file.')
    parser.add_argument('--metagene', required=True, help='Path to {name}_heatplot_start_diff.npy (Step 2 output).')
    parser.add_argument('--translon', required=True, help='Path to {name}_translon_dict.pickle (Step 1 output).')
    parser.add_argument('--symbol',   required=True, help='Path to symbol.csv (Step 1 output).')
    parser.add_argument('--output',   default='./transcript/', help='Output directory (default: ./transcript/)')
    parser.add_argument('--name',     required=True, help='Prefix for output file names.')
    parser.add_argument('--half-window', type=int, default=30,  help='Half-width of sliding window in nt (default: 30).')
    parser.add_argument('--meta-trim',   type=int, default=20,  help='Columns to trim from left of metagene (default: 20).')
    parser.add_argument('--rlen-min',    type=int, default=20,  help='Minimum read length row index (default: 20).')
    parser.add_argument('--rlen-max',    type=int, default=80,  help='Maximum read length row index, exclusive (default: 80).')
    parser.add_argument('--rlen-cap',    type=int, default=150, help='Read lengths >= this value are discarded (default: 150).')
    parser.add_argument('--min-reads',   type=int, default=10,  help='Minimum reads per transcript to process (default: 10).')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(
        bam_path      = args.bam,
        metagene_path = args.metagene,
        translon_path = args.translon,
        symbol_path   = args.symbol,
        output_dir    = args.output,
        fname         = args.name,
        half_window   = args.half_window,
        meta_trim     = args.meta_trim,
        rlen_min      = args.rlen_min,
        rlen_max      = args.rlen_max,
        rlen_cap      = args.rlen_cap,
        min_reads     = args.min_reads,
    )


if __name__ == '__main__':
    main()
