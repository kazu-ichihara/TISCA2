"""
TISCA2 Step 2 unit tests
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from tisca2.step2_metagene.metagene import build_metagene, parse_args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_read(positions, rlen):
    read = MagicMock()
    read.rlen = rlen
    read.get_reference_positions.return_value = positions
    return read

def _make_bam(reads_by_transcript):
    bam = MagicMock()
    bam.references = tuple(reads_by_transcript.keys())
    bam.fetch.side_effect = lambda name: reads_by_transcript.get(name, [])
    return bam


# ---------------------------------------------------------------------------
# build_metagene
# ---------------------------------------------------------------------------

class TestBuildMetagene:

    BASE = dict(
        bam_path   = 'dummy.bam',
        id_list    = ['TX1'],
        start_dict = {'TX1': 60},
        len_dict   = {'TX1': 200},
        upstream   = 50,
        downstream = 30,
        rlen_min   = 20,
        rlen_max   = 80,
        rlen_cap   = 90,
        min_reads  = 1,
    )

    def _run(self, reads, **overrides):
        args = {**self.BASE, **overrides}
        bam  = _make_bam({'TX1': reads})
        with patch('tisca2.step2_metagene.metagene.pysam.AlignmentFile', return_value=bam):
            return build_metagene(**args)

    def test_output_shape(self):
        metas = self._run([])
        assert metas.shape == (60, 80)

    def test_read_within_window_counted(self):
        read  = _make_read([60], rlen=30)
        metas = self._run([read])
        assert metas[30 - 20, 50] == 1

    def test_read_outside_rlen_cap_discarded(self):
        read  = _make_read([60], rlen=90)
        metas = self._run([read])
        assert metas.sum() == 0

    def test_min_reads_filter(self):
        read  = _make_read([60], rlen=30)
        metas = self._run([read], min_reads=10)
        assert metas.sum() == 0

    def test_window_boundary_skip_left(self):
        read  = _make_read([10], rlen=30)
        metas = self._run([read], start_dict={'TX1': 10})
        assert metas.sum() == 0

    def test_window_boundary_skip_right(self):
        read  = _make_read([185], rlen=30)
        metas = self._run([read], start_dict={'TX1': 185})
        assert metas.sum() == 0

    def test_missing_transcript_skipped(self):
        metas = self._run([], id_list=['UNKNOWN'])
        assert metas.sum() == 0

    def test_accumulates_multiple_reads(self):
        reads = [_make_read([60], rlen=30), _make_read([60], rlen=30)]
        metas = self._run(reads)
        assert metas[10, 50] == 2

    def test_differential_shape(self):
        read  = _make_read([60], rlen=30)
        metas = self._run([read])
        diff  = np.diff(metas, axis=1)
        assert diff.shape == (metas.shape[0], metas.shape[1] - 1)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:

    BASE = ['--bam', 'a.bam', '--ref', 'r.csv',
            '--symbol', 's.csv', '--name', 'x']

    def test_required_args(self):
        args = parse_args(self.BASE)
        assert args.bam  == 'a.bam'
        assert args.name == 'x'

    def test_defaults(self):
        args = parse_args(self.BASE)
        assert args.upstream   == 50
        assert args.downstream == 30
        assert args.rlen_min   == 20
        assert args.rlen_max   == 80
        assert args.rlen_cap   == 90
        assert args.min_reads  == 10

    def test_custom_window(self):
        args = parse_args(self.BASE + ['--upstream', '30', '--downstream', '20'])
        assert args.upstream   == 30
        assert args.downstream == 20

    def test_missing_required(self):
        with pytest.raises(SystemExit):
            parse_args(['--bam', 'a.bam'])
