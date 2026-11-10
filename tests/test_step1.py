"""
TISCA2 Step 1 unit tests
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tisca2.step1_translon.translon_generator import (
    find_orfs,
    build_position_map,
    build_annotated_translon,
    parse_args,
    run,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_SEQ = {
    # Minimal ORF: ATG start, TAG stop (pos 0-5, frame 0)
    'TX1': 'ATGAAATAG',
    # ORF in frame 1
    'TX2': 'XATGCTGTAA',  # X is invalid; not a codon -> frame 1: ATG at 1
}

SIMPLE_CODONS = ['ATG']

@pytest.fixture
def simple_orfs():
    return find_orfs({'TX1': 'ATGAAATAG'}, ['ATG'])

# ---------------------------------------------------------------------------
# find_orfs
# ---------------------------------------------------------------------------

class TestFindOrfs:
    def test_basic_atg(self, simple_orfs):
        assert len(simple_orfs) == 1
        row = simple_orfs.iloc[0]
        assert row['transcript ID'] == 'TX1'
        assert row['codon'] == 'ATG'
        assert row['start'] == 0
        assert row['stop'] == 6  # TAG は pos 6

    def test_no_stop_codon(self):
        # No stop codon -> no ORF
        df = find_orfs({'TX': 'ATGAAACCC'}, ['ATG'])
        assert len(df) == 0

    def test_multiple_codons(self):
        df = find_orfs({'TX': 'ATGAAATAACTGTAA'}, ['ATG', 'CTG'])
        codons_found = set(df['codon'])
        assert 'ATG' in codons_found
        assert 'CTG' in codons_found

    def test_multiple_frames(self):
        # frame 0: ATG at 0 -> stop at 6
        # frame 1: ATG at 1 -> stop somewhere
        seq = 'AATGTTTTAGATGCCC'  # deliberately mixed frames
        df = find_orfs({'TX': seq}, ['ATG'])
        frames_found = set((row['start'] % 3) for _, row in df.iterrows()
                           if row['transcript ID'] == 'TX')
        assert len(frames_found) >= 1

    def test_output_columns(self, simple_orfs):
        assert list(simple_orfs.columns) == ['transcript ID', 'codon', 'start', 'stop']

    def test_start_less_than_stop(self, simple_orfs):
        assert (simple_orfs['start'] < simple_orfs['stop']).all()

# ---------------------------------------------------------------------------
# build_position_map
# ---------------------------------------------------------------------------

class TestBuildPositionMap:
    def _make_fixtures(self):
        df = pd.DataFrame([
            {'transcript ID': 'TX1', 'codon': 'ATG', 'start': 0, 'stop': 6}
        ])
        # Exon: chr1:100-109 (plus strand)
        exon_dict = {'TX1': [(100, 109)]}
        d = pd.DataFrame(
            {'chr': ['chr1'], 'strand': ['+']},
            index=pd.Index(['TX1'], name='id')
        )
        return df, exon_dict, d

    def test_plus_strand_position(self):
        df, exon_dict, d = self._make_fixtures()
        pos = build_position_map(df, exon_dict, d)
        assert 'TX1_0' in pos
        assert pos['TX1_0'] == 'chr1:100'

    def test_missing_exon_skipped(self):
        df = pd.DataFrame([
            {'transcript ID': 'UNKNOWN', 'codon': 'ATG', 'start': 0, 'stop': 6}
        ])
        exon_dict = {}
        d = pd.DataFrame({'chr': [], 'strand': []})
        pos = build_position_map(df, exon_dict, d)
        assert len(pos) == 0

# ---------------------------------------------------------------------------
# build_annotated_translon
# ---------------------------------------------------------------------------

class TestBuildAnnotatedTranslon:
    """Tests for annotated CDS extraction using a minimal GTF file."""

    # Attribute order does not matter; parsed as key-value dict.
    GTF_LINES = '\n'.join([
        # transcript row
        'chr1\t.\ttranscript\t100\t200\t.\t+\t.\tgene_id "G1"; transcript_id "TX1"; gene_name "GeneA";',
        # exon row
        'chr1\t.\texon\t100\t200\t.\t+\t.\tgene_id "G1"; transcript_id "TX1"; gene_name "GeneA";',
        # start_codon: genomic position 110
        'chr1\t.\tstart_codon\t110\t112\t.\t+\t.\tgene_id "G1"; transcript_id "TX1"; gene_name "GeneA";',
        # stop_codon: genomic positions 130-132
        'chr1\t.\tstop_codon\t130\t132\t.\t+\t.\tgene_id "G1"; transcript_id "TX1"; gene_name "GeneA";',
    ])

    def _write_gtf(self, tmp_path):
        p = tmp_path / 'test.gtf'
        p.write_text(self.GTF_LINES)
        return str(p)

    def test_start_position(self, tmp_path):
        from tisca2.step1_translon.translon_generator import parse_gtf
        gtf_path = self._write_gtf(tmp_path)
        d, exon_dict = parse_gtf(gtf_path)
        ref, symbol = build_annotated_translon(gtf_path, exon_dict, d)
        # Genomic position 110 maps to transcript coordinate 10 on exon(100-200)
        assert 'TX1' in ref.index
        assert ref.loc['TX1', 'start'] == 10

    def test_symbol_extracted(self, tmp_path):
        from tisca2.step1_translon.translon_generator import parse_gtf
        gtf_path = self._write_gtf(tmp_path)
        d, exon_dict = parse_gtf(gtf_path)
        _, symbol = build_annotated_translon(gtf_path, exon_dict, d)
        assert 'TX1' in symbol.index
        assert symbol.loc['TX1', 'symbol'] == 'GeneA'
        # Verify that the length column is present
        assert 'length' in symbol.columns

    def test_missing_stop_excluded(self, tmp_path):
        """Transcripts without a stop_codon annotation should be excluded from ref."""
        lines = '\n'.join([
            'chr1\t.\ttranscript\t100\t200\t.\t+\t.\tgene_id "G2"; transcript_id "TX2"; extra "x"; gene_name "GeneB";',
            'chr1\t.\texon\t100\t200\t.\t+\t.\tgene_id "G2"; transcript_id "TX2"; extra "x"; gene_name "GeneB";',
            'chr1\t.\tstart_codon\t110\t112\t.\t+\t.\tgene_id "G2"; transcript_id "TX2"; extra "x"; gene_name "GeneB";',
        ])
        p = tmp_path / 'no_stop.gtf'
        p.write_text(lines)
        from tisca2.step1_translon.translon_generator import parse_gtf
        d, exon_dict = parse_gtf(str(p))
        ref, _ = build_annotated_translon(str(p), exon_dict, d)
        assert 'TX2' not in ref.index


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_required_args(self):
        args = parse_args(['--gtf', 'a.gtf', '--fasta', 'b.fa', '--name', 'test'])
        assert args.gtf == 'a.gtf'
        assert args.fasta == 'b.fa'
        assert args.name == 'test'

    def test_default_codons(self):
        args = parse_args(['--gtf', 'a.gtf', '--fasta', 'b.fa', '--name', 'test'])
        assert 'ATG' in args.codons

    def test_custom_codons(self):
        args = parse_args([
            '--gtf', 'a.gtf', '--fasta', 'b.fa', '--name', 'test',
            '--codons', 'ATG', 'CTG'
        ])
        assert args.codons == ['ATG', 'CTG']

    def test_missing_required(self):
        with pytest.raises(SystemExit):
            parse_args(['--fasta', 'b.fa', '--name', 'test'])  # --gtf missing
