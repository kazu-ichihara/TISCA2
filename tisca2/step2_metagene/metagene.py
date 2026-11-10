"""
TISCA2 - Step 2: Metagene Heatmap around CDS Start Sites

Accumulate a read-length-stratified coverage matrix around annotated CDS
start sites from a transcriptome-mapped BAM file and save the result and
its column-wise differential as .npy files for use in Step 3+4.

Usage:
    python -m tisca2.step2_metagene.metagene \
        --bam    mapped.bam \
        --ref    ./transcript/annotated_translon.csv \
        --symbol ./transcript/symbol.csv \
        --output ./transcript/ \
        --name   sample \
        [--upstream 50] [--downstream 30] \
        [--rlen-min 20] [--rlen-max 80] [--rlen-cap 90] \
        [--min-reads 10]

Outputs
-------
{name}_heatplot_start.npy      : Metagene coverage matrix
{name}_heatplot_start_diff.npy : Column-wise differential metagene matrix
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pysam


# ---------------------------------------------------------------------------
# Core accumulation
# ---------------------------------------------------------------------------

def build_metagene(
    bam_path: str,
    id_list: list,
    start_dict: dict,
    len_dict: dict,
    upstream: int   = 50,
    downstream: int = 30,
    rlen_min: int   = 20,
    rlen_max: int   = 80,
    rlen_cap: int   = 90,
    min_reads: int  = 10,
) -> np.ndarray:
    """
    Accumulate read-length-stratified coverage in a window around each
    annotated CDS start site.

    Parameters
    ----------
    bam_path : str
        Path to the BAM file (transcriptome-mapped, indexed).
    id_list : list
        Transcript IDs to process.
    start_dict : dict
        {transcript_id: CDS start position (0-based transcript coordinate)}
    len_dict : dict
        {transcript_id: transcript length (nt)}
    upstream : int
        Nucleotides upstream of start to include.
    downstream : int
        Nucleotides downstream of start to include.
    rlen_min : int
        Minimum read length row index in the output matrix.
    rlen_max : int
        Maximum read length row index, exclusive.
    rlen_cap : int
        Read lengths >= rlen_cap are discarded.
    min_reads : int
        Minimum number of reads required to process a transcript.

    Returns
    -------
    metas : np.ndarray, shape (rlen_max - rlen_min, upstream + downstream)
    """
    n_rows = rlen_max - rlen_min
    n_cols = upstream + downstream
    metas  = np.zeros((n_rows, n_cols))

    bamfile = pysam.AlignmentFile(bam_path, 'rb')
    accept = list(set(bamfile.references)&set(id_list))
    for name in accept:
        if name not in start_dict or name not in len_dict:
            continue

        start  = start_dict[name]
        length = len_dict[name]

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

        if n_reads < min_reads:
            continue

        col_start = start - upstream
        col_end   = start + downstream
        if col_start < 0 or col_end > length:
            continue

        window = m[rlen_min:rlen_max, col_start:col_end]
        if window.shape == (n_rows, n_cols):
            metas += window

    bamfile.close()
    return metas


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    bam_path: str,
    ref_path: str,
    symbol_path: str,
    output_dir: str,
    fname: str,
    upstream: int,
    downstream: int,
    rlen_min: int,
    rlen_max: int,
    rlen_cap: int,
    min_reads: int,
) -> None:
    """Run Step 2: build and save the metagene heatmap."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'[Step 2] Loading annotated translon: {ref_path}')
    ref        = pd.read_csv(ref_path, index_col=0)
    start_dict = ref['start'].to_dict()

    print(f'[Step 2] Loading symbol table: {symbol_path}')
    symbol   = pd.read_csv(symbol_path, index_col=0)
    len_dict = symbol['length'].to_dict()

    id_list = list(ref.index)
    print(f'[Step 2] Transcripts to process: {len(id_list):,}')
    print(f'[Step 2] Accumulating metagene from BAM: {bam_path}')

    metas = build_metagene(
        bam_path   = bam_path,
        id_list    = id_list,
        start_dict = start_dict,
        len_dict   = len_dict,
        upstream   = upstream,
        downstream = downstream,
        rlen_min   = rlen_min,
        rlen_max   = rlen_max,
        rlen_cap   = rlen_cap,
        min_reads  = min_reads,
    )

    out_path = out / f'{fname}_heatplot_start'
    np.save(str(out_path), metas)
    print(f'[Step 2] Saved metagene matrix: {out_path}.npy  shape={metas.shape}')

    diff      = np.diff(metas, axis=1)
    diff_path = out / f'{fname}_heatplot_start_diff'
    np.save(str(diff_path), diff)
    print(f'[Step 2] Saved differential metagene: {diff_path}.npy  shape={diff.shape}')

    print('[Step 2] Done.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='tisca2-step2',
        description='TISCA2 Step 2: Build a metagene heatmap around CDS start sites.'
    )
    parser.add_argument('--bam',    required=True, help='Path to the transcriptome-mapped BAM file.')
    parser.add_argument('--ref',    required=True, help='Path to annotated_translon.csv (Step 1 output).')
    parser.add_argument('--symbol', required=True, help='Path to symbol.csv (Step 1 output).')
    parser.add_argument('--output', default='./transcript/', help='Output directory (default: ./transcript/)')
    parser.add_argument('--name',   required=True, help='Prefix for output file names.')
    parser.add_argument('--upstream',   type=int, default=50, help='Nucleotides upstream of CDS start (default: 50).')
    parser.add_argument('--downstream', type=int, default=30, help='Nucleotides downstream of CDS start (default: 30).')
    parser.add_argument('--rlen-min',   type=int, default=20, help='Minimum read length row index (default: 20).')
    parser.add_argument('--rlen-max',   type=int, default=80, help='Maximum read length row index, exclusive (default: 80).')
    parser.add_argument('--rlen-cap',   type=int, default=90, help='Read lengths >= this value are discarded (default: 90).')
    parser.add_argument('--min-reads',  type=int, default=10, help='Minimum reads per transcript to process (default: 10).')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(
        bam_path    = args.bam,
        ref_path    = args.ref,
        symbol_path = args.symbol,
        output_dir  = args.output,
        fname       = args.name,
        upstream    = args.upstream,
        downstream  = args.downstream,
        rlen_min    = args.rlen_min,
        rlen_max    = args.rlen_max,
        rlen_cap    = args.rlen_cap,
        min_reads   = args.min_reads,
    )


if __name__ == '__main__':
    main()
