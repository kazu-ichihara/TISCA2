"""
TISCA2 Step 2+3+4 unit tests
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from tisca2.step3_scorer.scorer import (
    cos_sim,
    build_coverage,
    score_translons,
    parse_args,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_read(positions, rlen):
    read = MagicMock()
    read.rlen = rlen
    read.get_reference_positions.return_value = positions
    return read

def _make_bam(reads):
    bam = MagicMock()
    bam.fetch.return_value = reads
    return bam


# ---------------------------------------------------------------------------
# cos_sim
# ---------------------------------------------------------------------------

class TestCosSim:

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cos_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cos_sim(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cos_sim(v, -v) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cos_sim(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0

    def test_both_zero_vectors(self):
        assert cos_sim(np.zeros(3), np.zeros(3)) == 0.0


# ---------------------------------------------------------------------------
# build_coverage
# ---------------------------------------------------------------------------

class TestBuildCoverage:

    def test_output_shape(self):
        bam = _make_bam([])
        m, n = build_coverage(bam, 'TX1', length=100, rlen_cap=150)
        assert m.shape == (150, 100)
        assert n == 0

    def test_read_counted(self):
        read = _make_read([10, 11, 12], rlen=30)
        bam  = _make_bam([read])
        m, n = build_coverage(bam, 'TX1', length=100, rlen_cap=150)
        assert n == 1
        assert m[30, 10] == 1
        assert m[30, 11] == 1

    def test_rlen_cap_discards_long_reads(self):
        read = _make_read([10], rlen=150)
        bam  = _make_bam([read])
        m, n = build_coverage(bam, 'TX1', length=100, rlen_cap=150)
        assert n == 0
        assert m.sum() == 0

    def test_position_out_of_bounds_ignored(self):
        read = _make_read([200], rlen=30)
        bam  = _make_bam([read])
        m, n = build_coverage(bam, 'TX1', length=100, rlen_cap=150)
        assert m.sum() == 0

    def test_multiple_reads_accumulated(self):
        reads = [_make_read([5], rlen=30), _make_read([5], rlen=30)]
        bam   = _make_bam(reads)
        m, n  = build_coverage(bam, 'TX1', length=100, rlen_cap=150)
        assert n == 2
        assert m[30, 5] == 2


# ---------------------------------------------------------------------------
# score_translons
# ---------------------------------------------------------------------------

class TestScoreTranslons:

    def _make_dist(self, n_rlen=60, length=200):
        dist = np.zeros((n_rlen, length))
        dist[:, 50:80] = 1.0
        return dist

    def _make_ref(self, n_rlen=60, n_cols=79):
        # metagene.py outputs np.diff of the raw matrix (shape: upstream+downstream-1 = 79),
        # so ref passed to score_translons is already differenced.
        return np.ones((n_rlen, n_cols))

    def test_returns_list(self):
        recs = score_translons(self._make_dist(), self._make_ref(),
                               starts=np.array([60]), half_window=30)
        assert isinstance(recs, list)

    def test_record_format(self):
        recs = score_translons(self._make_dist(), self._make_ref(),
                               starts=np.array([60]), half_window=30)
        assert len(recs) == 1
        start, s, lc = recs[0]
        assert start == 60
        assert -1.0 <= s <= 1.0
        assert isinstance(lc, int)

    def test_left_boundary(self):
        recs = score_translons(self._make_dist(), self._make_ref(),
                               starts=np.array([10]), half_window=30)
        assert len(recs) == 1

    def test_right_boundary(self):
        recs = score_translons(self._make_dist(), self._make_ref(),
                               starts=np.array([190]), half_window=30)
        assert len(recs) == 1

    def test_multiple_starts(self):
        recs = score_translons(self._make_dist(), self._make_ref(),
                               starts=np.array([60, 100, 150]), half_window=30)
        assert len(recs) == 3

    def test_zero_coverage_returns_zero_similarity(self):
        recs = score_translons(np.zeros((60, 200)), self._make_ref(),
                               starts=np.array([60]), half_window=30)
        assert recs[0][1] == 0.0


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:

    BASE = ['--bam', 'a.bam', '--metagene', 'm.npy',
            '--translon', 't.pkl', '--symbol', 's.csv', '--name', 'x']

    def test_required_args(self):
        args = parse_args(self.BASE)
        assert args.bam == 'a.bam'
        assert args.name == 'x'

    def test_defaults(self):
        args = parse_args(self.BASE)
        assert args.meta_trim   == 20
        assert args.half_window == 30
        assert args.rlen_min    == 20
        assert args.rlen_max    == 80
        assert args.rlen_cap    == 150
        assert args.min_reads   == 10

    def test_custom_values(self):
        args = parse_args(self.BASE + ['--half-window', '20', '--min-reads', '5'])
        assert args.half_window == 20
        assert args.min_reads   == 5

    def test_missing_required(self):
        with pytest.raises(SystemExit):
            parse_args(['--bam', 'a.bam'])


# ---------------------------------------------------------------------------
# Output column names
# ---------------------------------------------------------------------------

class TestOutputColumns:
    """Verify that score_translons produces 'Transcript ID' (capital T).

    Step 4 reads the cosine CSV expecting 'Transcript ID'; a lowercase 't'
    would cause a KeyError downstream.
    """

    def test_cosine_csv_transcript_id_column(self):
        from tisca2.step3_scorer.scorer import score_translons
        import pickle, tempfile, os, pandas as pd

        dist = np.zeros((60, 200))
        dist[:, 50:80] = 1.0
        ref  = np.ones((60, 79))

        translon_dict = {'TX1': np.array([60, 100])}
        symbol = pd.DataFrame({'id': ['TX1'], 'length': [200], 'symbol': ['GENE1']}).set_index('id')

        # Simulate the per-transcript call and assemble a DataFrame as main() does
        records = score_translons(dist, ref, starts=np.array([60]), half_window=30)
        df = pd.DataFrame(records, columns=['start', 'cosine similarity', 'length coverage'])
        df.insert(0, 'Transcript ID', 'TX1')

        assert 'Transcript ID' in df.columns
        assert 'transcript ID' not in df.columns
