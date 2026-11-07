"""
TISCA2 - Step 1: Translon List Generator

Create a translon list from a genome GTF file and transcriptome FASTA file.

Usage:
    python -m tisca2.step1_translon.translon_generator \
        --gtf ./ref/annotation.gtf \
        --fasta ./ref/transcript.fasta \
        --output ./transcript/ \
        --name sample \
        [--codons ATG CTG GTG ...]

Outputs
-------
{name}_translon.csv          : All ORF (translon) records
{name}_translon_dict.pickle  : {transcript_id: start_positions[]} dictionary
{name}_translon_position.csv : Genomic coordinate map for each translon start
annotated_translon.csv       : Annotated CDS coordinates in transcript space (from GTF)
symbol.csv                   : transcript_id -> gene symbol mapping
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

# Default start codons
DEFAULT_CODONS = ['ATG', 'CTG', 'GTG', 'ACG', 'TTG', 'ATC', 'ATA', 'AGG', 'AAG', 'ATT']
STOP_CODONS    = ['TAG', 'TGA', 'TAA']


# ---------------------------------------------------------------------------
# GTF parsing
# ---------------------------------------------------------------------------

def parse_gtf(gtf_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Parse a GTF file and return a chr/strand DataFrame and an exon dictionary.

    Parameters
    ----------
    gtf_path : str
        Path to the GTF annotation file.

    Returns
    -------
    d : pd.DataFrame
        index=transcript_id, columns=['chr', 'strand']
    exon_dict : dict
        {transcript_id: [(genomic_start, genomic_end), ...]}
        Minus-strand transcripts are sorted in descending order (transcript coordinate order).
    """
    gtf = pd.read_csv(gtf_path, sep='\t', header=None, comment='#')
    gtf = gtf.drop([1, 5, 7], axis=1)
    gtf.columns = ['chr', 'type', 'start', 'end', 'strand', 'annotation']
    gtf['id'] = [x.split(';')[1].split('"')[1] for x in gtf['annotation']]

    # chr/strand lookup table per transcript
    d = gtf[['chr', 'strand', 'id']].drop_duplicates('id').set_index('id')

    # Exon dictionary
    gtf_exon = gtf[gtf['type'] == 'exon'][['chr', 'start', 'end', 'strand', 'id']]
    ex_plus  = gtf_exon[gtf_exon['strand'] == '+']
    ex_minus = gtf_exon[gtf_exon['strand'] != '+'].sort_values('start', ascending=False)

    r_plus = (
        ex_plus.groupby('id', sort=False)
        .apply(lambda x: list(zip(x['start'], x['end'])))
        .to_dict()
    )
    r_minus = (
        ex_minus.groupby('id', sort=False)
        .apply(lambda x: list(zip(x['start'], x['end'])))
        .to_dict()
    )
    exon_dict = {**r_plus, **r_minus}

    return d, exon_dict


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta(fasta_path: str) -> dict:
    """
    Parse a transcriptome FASTA file and return {transcript_id: sequence}.
    The transcript ID is taken as the first field before '|' in the record ID.
    """
    seq_dict = {}
    for record in SeqIO.parse(fasta_path, 'fasta'):
        name = record.id.split('|')[0]
        seq_dict[name] = str(record.seq).upper()
    return seq_dict


# ---------------------------------------------------------------------------
# ORF detection
# ---------------------------------------------------------------------------

def find_orfs(seq_dict: dict, codons: list[str]) -> pd.DataFrame:
    """
    Detect all ORFs (translons) across all transcripts and reading frames.

    Parameters
    ----------
    seq_dict : dict
        {transcript_id: sequence}
    codons : list[str]
        Start codons to search for.

    Returns
    -------
    df : pd.DataFrame
        columns=['Transcript ID', 'codon', 'start', 'stop']
        start/stop are 0-based nucleotide positions on the transcript.
    """
    orfs = []
    for name, seq in seq_dict.items():
        for frame in [0, 1, 2]:
            s1 = seq[frame:]
            ss = np.array([s1[i:i+3] for i in range(0, len(s1) + 1, 3)])

            # Stop codon mask
            stops = np.sum(
                [np.where(ss == cod, 1, 0) for cod in STOP_CODONS],
                axis=0
            )

            for codon in codons:
                starts = np.where(ss == codon, 1, 0)
                pos = np.where(starts > 0)[0]
                for p in pos:
                    stop_rel = np.where(stops[p:] > 0)[0]
                    if len(stop_rel) > 0:
                        p1 = frame + p * 3
                        p2 = frame + (p + stop_rel[0]) * 3
                        orfs.append([name, codon, p1, p2])

    df = pd.DataFrame(orfs, columns=['Transcript ID', 'codon', 'start', 'stop'])
    return df


# ---------------------------------------------------------------------------
# Genomic coordinate mapping
# ---------------------------------------------------------------------------

def build_position_map(df: pd.DataFrame, exon_dict: dict, d: pd.DataFrame) -> dict:
    """
    Convert translon start positions from transcript coordinates to genomic coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        Output of find_orfs().
    exon_dict : dict
        Output of parse_gtf().
    d : pd.DataFrame
        Output of parse_gtf() (chr/strand information).

    Returns
    -------
    pos_dict : dict
        {'{transcript_id}_{start}': '{chrom}:{genomic_pos}'}
    """
    d1 = {k: np.array(v['start']) for k, v in df.groupby('Transcript ID')}
    pos_dict = {}

    for name, starts in d1.items():
        if name not in exon_dict:
            continue
        ex = exon_dict[name]
        chrom  = d.loc[name, 'chr']
        strand = d.loc[name, 'strand']

        # Expand exon regions in transcript coordinate order
        exon_coords = []
        if strand == '+':
            for e in ex:
                exon_coords.append(np.arange(e[0], e[1] + 1, 1))
        else:
            for e in ex:
                exon_coords.append(np.arange(e[1], e[0] - 1, -1))
        exon_arr = np.concatenate(exon_coords)

        for start in starts:
            if start < len(exon_arr):
                genomic_pos = exon_arr[start]
                pos_dict[f'{name}_{start}'] = f'{chrom}:{genomic_pos}'

    return pos_dict


# ---------------------------------------------------------------------------
# Annotated CDS coordinate construction
# ---------------------------------------------------------------------------

def build_annotated_translon(gtf_path: str, exon_dict: dict, d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract known CDS transcript coordinates from GTF start_codon/stop_codon entries.

    Parameters
    ----------
    gtf_path : str
        Path to the GTF annotation file.
    exon_dict : dict
        Output of parse_gtf().
    d : pd.DataFrame
        Output of parse_gtf() (chr/strand information).

    Returns
    -------
    ref : pd.DataFrame
        index=transcript_id, columns=['start', 'stop']
        0-based nucleotide positions on the transcript.
    symbol : pd.DataFrame
        index=transcript_id, columns=['length', 'symbol']
    """
    # Exclude gene-level rows; work at transcript level
    gtf_raw = pd.read_csv(gtf_path, sep='\t', header=None, comment='#')
    gtf_raw = gtf_raw.drop([1, 5, 7], axis=1)
    gtf_raw.columns = ['chr', 'type', 'start', 'end', 'strand', 'annotation']
    gtf_tr = gtf_raw[gtf_raw['type'] != 'gene'].copy()
    gtf_tr['id']     = [x.split(';')[1].split('"')[1] for x in gtf_tr['annotation']]
    gtf_tr['symbol'] = [x.split(';')[3].split('"')[1] for x in gtf_tr['annotation']]

    # Gene symbol table (also store transcript length)
    gtf_tr['length'] = gtf_tr['end'] - gtf_tr['start']
    symbol = gtf_tr[['id', 'length', 'symbol']].drop_duplicates('id').set_index('id')

    # Keep only transcripts that have both start_codon and stop_codon annotations
    gtf_codon = gtf_tr[gtf_tr['type'].isin(['start_codon', 'stop_codon'])]
    codon_types = gtf_codon.groupby('id')['type'].apply(lambda x: ' '.join(x.unique()))
    valid_ids = list(
        codon_types[
            codon_types.str.contains('start_codon') & codon_types.str.contains('stop_codon')
        ].index
    )
    gtf_coding = gtf_tr[gtf_tr['id'].isin(valid_ids)]

    gtf_start = gtf_coding[gtf_coding['type'] == 'start_codon']
    gtf_stop  = gtf_coding[gtf_coding['type'] == 'stop_codon']

    start_dict: dict = {}
    stop_dict: dict  = {}

    for name in valid_ids:
        if name not in exon_dict:
            continue

        starts_g = gtf_start[gtf_start['id'] == name][['start', 'end']].values
        stops_g  = gtf_stop[gtf_stop['id'] == name][['start', 'end']].values
        if len(starts_g) == 0 or len(stops_g) == 0:
            continue

        ex     = exon_dict[name]  # list of (genomic_start, genomic_end)
        strand = d.loc[name, 'strand']

        exon_coords = []
        if strand == '+':
            for e in ex:
                exon_coords.append(np.arange(e[0], e[1] + 1))
            pos1 = starts_g[0][0]   # start_codon genomic start
            pos2 = stops_g[-1][-1]  # stop_codon genomic end
        else:
            for e in ex:
                exon_coords.append(np.arange(e[1], e[0] - 1, -1))
            pos1 = starts_g[0][-1]  # minus strand: start_codon genomic end
            pos2 = stops_g[-1][0]   # minus strand: stop_codon genomic start

        exon_arr = np.concatenate(exon_coords)

        idx1 = np.where(exon_arr == pos1)[0]
        idx2 = np.where(exon_arr == pos2)[0]
        if len(idx1) == 0 or len(idx2) == 0:
            continue

        start_dict[name] = int(idx1[0])
        stop_dict[name]  = int(idx2[0]) + 1

    ref = pd.DataFrame({'start': start_dict, 'stop': stop_dict})
    return ref, symbol


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(gtf_path: str, fasta_path: str, output_dir: str, fname: str, codons: list[str]) -> None:
    """Run Step 1: generate translon list and annotated CDS coordinates."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[Step 1] Loading GTF: {gtf_path}")
    d, exon_dict = parse_gtf(gtf_path)

    print(f"[Step 1] Loading FASTA: {fasta_path}")
    seq_dict = parse_fasta(fasta_path)

    # Annotated CDS
    print("[Step 1] Building annotated CDS coordinates ...")
    ref, symbol = build_annotated_translon(gtf_path, exon_dict, d)
    ref_csv    = out / 'annotated_translon.csv'
    symbol_csv = out / 'symbol.csv'
    ref.to_csv(ref_csv)
    symbol.to_csv(symbol_csv)
    print(f"[Step 1] Saved annotated CDS: {ref_csv}  ({len(ref):,} records)")
    print(f"[Step 1] Saved gene symbols:  {symbol_csv}  ({len(symbol):,} records)")

    # All ORF search
    print(f"[Step 1] Searching ORFs (codons: {codons}) ...")
    df = find_orfs(seq_dict, codons)
    translon_csv = out / f'{fname}_translon.csv'
    df.to_csv(translon_csv, index=False)
    print(f"[Step 1] Saved translon list: {translon_csv}  ({len(df):,} records)")

    # Translon dictionary (pickle)
    d1 = {k: np.array(v['start']) for k, v in df.groupby('Transcript ID')}
    translon_pkl = out / f'{fname}_translon_dict.pickle'
    with open(translon_pkl, 'wb') as fh:
        pickle.dump(d1, fh)
    print(f"[Step 1] Saved translon dict: {translon_pkl}")

    # Genomic coordinate map
    print("[Step 1] Converting to genomic coordinates ...")
    pos_dict = build_position_map(df, exon_dict, d)
    pos_csv = out / f'{fname}_translon_position.csv'
    pd.Series(pos_dict).to_csv(pos_csv, header=['genomic_position'])
    print(f"[Step 1] Saved position map: {pos_csv}  ({len(pos_dict):,} records)")

    print("[Step 1] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='tisca2-step1',
        description='TISCA2 Step 1: Create a translon list from GTF and FASTA files.'
    )
    parser.add_argument(
        '--gtf', required=True,
        help='Path to the genome annotation GTF file.'
    )
    parser.add_argument(
        '--fasta', required=True,
        help='Path to the transcriptome FASTA file.'
    )
    parser.add_argument(
        '--output', default='./transcript/',
        help='Output directory (default: ./transcript/)'
    )
    parser.add_argument(
        '--name', required=True,
        help='Prefix for output file names (e.g. sample)'
    )
    parser.add_argument(
        '--codons', nargs='+', default=DEFAULT_CODONS,
        metavar='CODON',
        help=(
            'Start codons to search for (space-separated). '
            f'Default: {" ".join(DEFAULT_CODONS)}'
        )
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    codons = [c.upper() for c in args.codons]
    run(
        gtf_path   = args.gtf,
        fasta_path = args.fasta,
        output_dir = args.output,
        fname      = args.name,
        codons     = codons,
    )


if __name__ == '__main__':
    main()
