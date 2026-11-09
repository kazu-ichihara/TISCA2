"""
TISCA2 - Step 4: Filtering, Classification, and Output Formatting

Filter translon candidates by cosine similarity threshold (derived from a
precision-recall curve against annotated CDS), retain local maxima per
transcript, classify each translon relative to the annotated CDS, and write
the final results table.

Usage:
    python -m tisca2.step4_filter.filter_classify \
        --fasta      ./ref/transcript.fasta \
        --cosine     ./transcript/{name}_cosine.csv \
        --position   ./transcript/{name}_translon_position.csv \
        --symbol     ./transcript/symbol.csv \
        --ref        ./transcript/annotated_translon.csv \
        --output     ./transcript/ \
        --name       sample \
        [--min-length-coverage 10] \
        [--novel-prefix MSTRG]

Outputs
-------
{name}_cosine_eff.csv   : Filtered cosine scores with position and symbol columns
{name}_PR_curve.png     : Precision-recall curve
{name}_PR_threshold.png : PR / F1 curves with threshold line
{name}_results.csv      : Final classified translon table
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.signal import argrelmax
from sklearn import metrics
from sklearn.metrics import precision_recall_curve


# ---------------------------------------------------------------------------
# FASTA loading
# ---------------------------------------------------------------------------

def load_fasta(fasta_path: str) -> dict:
    """
    Load transcriptome FASTA and return {transcript_id: Seq} dict.
    Seq objects are kept (not converted to str) to allow .translate().
    """
    seq_dict = {}
    for record in SeqIO.parse(fasta_path, 'fasta'):
        name = record.id.split('|')[0]
        seq_dict[name] = record.seq
    return seq_dict


# ---------------------------------------------------------------------------
# PR curve and threshold
# ---------------------------------------------------------------------------

def compute_pr_threshold(
    sc: pd.DataFrame,
    ref_index: pd.Index,
    starts: pd.Index,
    output_dir: Path,
    fname: str,
) -> float:
    """
    Compute precision-recall curve against annotated CDS starts and find the
    threshold where precision == recall (interpolated crossing point).

    Annotated CDS starts that are absent from the scored set are added as
    false negatives with a fixed score of -0.5.

    Parameters
    ----------
    sc : pd.DataFrame
        Filtered cosine score table (index = 'TIS ID').
    ref_index : pd.Index
        Transcript IDs present in annotated_translon.csv.
    starts : pd.Index
        '{transcript_id}_{start}' labels of all annotated CDS starts.
    output_dir : Path
    fname : str

    Returns
    -------
    thres : float
        Cosine similarity threshold at the precision==recall crossing.
    """
    sc_test = sc[sc['transcript ID'].isin(ref_index)][['cosine similarity']].copy()
    sc_test['true'] = sc_test.index.isin(starts).astype(int)

    # Add annotated starts missing from the scored set as hard negatives
    neg = list(set(starts) - set(sc_test.index))
    y_true   = list(sc_test['true'])   + [1] * len(neg)
    y_scores = list(sc_test['cosine similarity']) + [-0.5] * len(neg)
    y_true   = np.array(y_true)
    y_scores = np.array(y_scores)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auc = metrics.auc(recall[1:], precision[1:])

    # PR curve plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall[1:], precision[1:])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    fig.savefig(str(output_dir / f'{fname}_PR_curve.png'), dpi=300)
    plt.close(fig)

    # F1 and threshold plot
    df = pd.DataFrame({
        'Precision':  precision,
        'Recall':     recall,
        'Thresholds': np.append(thresholds, 0),
    })[1:-1]
    df['F1'] = 2 * df['Precision'] * df['Recall'] / (df['Precision'] + df['Recall'])
    x  = df['Thresholds']
    y1 = df['Precision']
    y2 = df['Recall']
    y3 = df['F1']

    # Interpolate crossing of precision and recall curves
    idx = np.where(np.diff(np.sign(y1.values - y2.values)))[0]
    thres = float(thresholds[-1])  # fallback
    for i in idx:
        x0, x1   = x.iloc[i], x.iloc[i + 1]
        y0, y1_  = y1.iloc[i] - y2.iloc[i], y1.iloc[i + 1] - y2.iloc[i + 1]
        if y1_ != y0:
            thres = x0 - y0 * (x1 - x0) / (y1_ - y0)

    print(f'AUC: {auc:.4f}')
    print(f'Threshold: {thres:.4f}')

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x, y1, label='Precision')
    ax.plot(x, y2, label='Recall')
    ax.plot(x, y3, label='F1')
    ax.set_ylim(0, 1)
    ax.axvline(thres, linestyle='dashed', color='black', linewidth=1)
    fig.savefig(str(output_dir / f'{fname}_PR_threshold.png'), dpi=300)
    plt.close(fig)

    return thres


# ---------------------------------------------------------------------------
# Local maxima filter
# ---------------------------------------------------------------------------

def find_local_maxima(sc: pd.DataFrame) -> list:
    """
    For each transcript, retain translon indices that are local maxima of
    cosine similarity. A sentinel value of -1 is prepended so the first
    element can also be selected as a local maximum.
    """
    lmax = []
    for _tid, group in sc.groupby('transcript ID'):
        y = group['cosine similarity'].values
        y = np.insert(y, 0, -1)
        a = argrelmax(y)[0] - 1
        lmax.extend(group.index[a])
    return lmax


# ---------------------------------------------------------------------------
# ORF type classification
# ---------------------------------------------------------------------------

def classify_orf_type(
    start: int,
    stop: int,          # exclusive stop (start + ORF_length*3 + 3)
    cstart: int,        # annotated CDS start (-1 if noncoding)
    cstop: int,         # annotated CDS stop
    out_of_frame: bool,
    name: str,
    novel_prefix: str,
) -> str:
    """
    Classify a translon relative to the annotated CDS.

    Parameters
    ----------
    start : int
        Translon start (transcript coordinate).
    stop : int
        Translon stop, inclusive of stop codon (start + ORF_length*3 + 3).
    cstart : int
        Annotated CDS start (-1 for noncoding transcripts).
    cstop : int
        Annotated CDS stop.
    out_of_frame : bool
        True if the translon is out of frame with the annotated CDS.
    name : str
        Transcript ID.
    novel_prefix : str
        Transcript ID prefix that marks novel/unannotated transcripts.

    Returns
    -------
    orf_type : str
    """
    if cstart < 0:  # noncoding transcript
        return 'novel' if name.startswith(novel_prefix) else 'noncoding'

    if out_of_frame:
        if start < cstart:  # upstream
            if stop < cstart:
                return 'uORF'
            elif stop < cstop:
                return 'overlap.uORF'
            else:
                return 'polycistronic'
        else:               # downstream
            if stop <= cstop:
                return 'internal'
            elif start < cstop:
                return 'external'
            else:
                return 'polycistronic'
    else:           # in-frame
        if start == cstart:
            return 'canonical' if stop == cstop else 'internal'
        elif stop == cstop:
            return 'extension' if start < cstart else 'truncation'
        else:
            if start < cstart:
                return 'overlap.uORF' if stop >= cstart else 'uORF'
            else:
                if stop < cstop:
                    return 'internal'
                else:
                    return 'polycistronic'


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    fasta_path: str,
    cosine_path: str,
    position_path: str,
    symbol_path: str,
    ref_path: str,
    output_dir: str,
    fname: str,
    min_length_coverage: int,
    novel_prefix: str,
    threshold: float | None = None,
) -> None:
    """Run Step 4: filter, threshold, classify, and save results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'[Step 4] Loading FASTA: {fasta_path}')
    seq_dict = load_fasta(fasta_path)

    print(f'[Step 4] Loading symbol table: {symbol_path}')
    symbol      = pd.read_csv(symbol_path, index_col=0)
    symbol_dict = symbol['symbol'].to_dict()
    len_dict    = symbol['length'].to_dict()

    print(f'[Step 4] Loading annotated translon: {ref_path}')
    ref        = pd.read_csv(ref_path, index_col=0)
    start_dict = ref['start'].to_dict()
    stop_dict  = ref['stop'].to_dict()

    print(f'[Step 4] Loading cosine scores: {cosine_path}')
    sc = pd.read_csv(cosine_path)
    sc = sc[sc['length coverage'] >= min_length_coverage]
    sc.index      = sc['transcript ID'] + '_' + sc['start'].astype(str)
    sc.index.name = 'TIS ID'
    sc = sc.sort_values(['transcript ID', 'start'])

    position = pd.read_csv(position_path, index_col=0, header=None)[1]
    sc['position'] = position.reindex(sc.index)
    sc['symbol']   = [symbol_dict[x] for x in sc['transcript ID']]
    sc.to_csv(out / f'{fname}_cosine_eff.csv')
    print(f'[Step 4] Saved filtered scores: {out / f"{fname}_cosine_eff.csv"}  ({len(sc):,} records)')

    # Threshold: use CLI value if provided, otherwise derive from PR curve
    starts = ref.index + '_' + ref['start'].astype(str)
    if threshold is not None:
        thres = threshold
        print(f'[Step 4] Using user-specified threshold: {thres}')
    else:
        print('[Step 4] Computing PR curve and threshold ...')
        thres = compute_pr_threshold(sc, ref.index, starts, out, fname)

    # Local maxima filter
    lmax = find_local_maxima(sc)
    sc_  = sc[(sc['cosine similarity'] >= thres) & (sc.index.isin(lmax))].copy()
    print(f'[Step 4] Candidates after threshold + local maxima filter: {len(sc_):,}')

    # ORF annotation
    canonical_list, seq_list, codon_list = [], [], []
    for i in range(len(sc_)):
        name  = sc_['transcript ID'].values[i]
        start = sc_['start'].values[i]
        seq_list.append(str(seq_dict[name][start:].translate(to_stop=True)))
        codon_list.append(str(seq_dict[name][start:start + 3]).upper())
        tis = start_dict.get(name)
        canonical_list.append(tis if tis is not None else -1)

    sc_['canonical']          = canonical_list
    sc_['seq']                = seq_list
    sc_['codon']              = codon_list
    sc_['ORF_length']         = sc_['seq'].apply(len)
    sc_['stop']               = sc_['start'] + sc_['ORF_length'] * 3
    sc_['transcript_length']  = [len_dict[x] for x in sc_['transcript ID']]

    # Out-of-frame flag
    out_of_frame = []
    for i in range(len(sc_)):
        c = sc_['canonical'].values[i]
        if c >= 0:
            out_of_frame.append(c % 3 != sc_['start'].values[i] % 3)
        else:
            out_of_frame.append(True)
    sc_['out'] = out_of_frame

    # ORF type classification
    types = []
    for i in range(len(sc_)):
        name   = sc_['transcript ID'].values[i]
        start  = sc_['start'].values[i]
        stop   = sc_['stop'].values[i] + 3
        length = len_dict[name]
        cstart = sc_['canonical'].values[i]
        cstop  = stop_dict.get(name, length)
        orf_type = classify_orf_type(
            start, stop, cstart, cstop,
            sc_['out'].values[i], name, novel_prefix,
        )
        types.append(orf_type)
    sc_['type'] = types

    results_csv = out / f'{fname}_results.csv'
    sc_.to_csv(results_csv)
    print(f'[Step 4] Saved final results: {results_csv}  ({len(sc_):,} records)')
    print('[Step 4] Done.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="tisca2-step4",
        description='TISCA2 Step 4: Filter, classify, and format translon results.'
    )
    parser.add_argument('--fasta',    required=True, help='Path to transcriptome FASTA (Step 1 input).')
    parser.add_argument('--cosine',   required=True, help='Path to {name}_cosine.csv (Step 3+4 output).')
    parser.add_argument('--position', required=True, help='Path to {name}_translon_position.csv (Step 1 output).')
    parser.add_argument('--symbol',   required=True, help='Path to symbol.csv (Step 1 output).')
    parser.add_argument('--ref',      required=True, help='Path to annotated_translon.csv (Step 1 output).')
    parser.add_argument('--output',   default='./transcript/', help='Output directory (default: ./transcript/)')
    parser.add_argument('--name',     required=True, help='Prefix for output file names.')
    parser.add_argument('--min-length-coverage', type=int, default=10,
                        help='Minimum length coverage to retain a translon (default: 10).')
    parser.add_argument('--novel-prefix', default='MSTRG',
                        help='Transcript ID prefix marking novel transcripts (default: MSTRG).')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Cosine similarity threshold for filtering. '
                             'If omitted, the threshold is derived from the PR curve '
                             '(precision == recall crossing point).')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(
        fasta_path          = args.fasta,
        cosine_path         = args.cosine,
        position_path       = args.position,
        symbol_path         = args.symbol,
        ref_path            = args.ref,
        output_dir          = args.output,
        fname               = args.name,
        min_length_coverage = args.min_length_coverage,
        novel_prefix        = args.novel_prefix,
        threshold           = args.threshold,
    )


if __name__ == '__main__':
    main()
