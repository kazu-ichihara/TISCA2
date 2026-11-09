"""
TISCA2 Step 4 unit tests
"""

import numpy as np
import pandas as pd
import pytest

from tisca2.step4_filter.filter_classify import (
    classify_orf_type,
    find_local_maxima,
    parse_args,
)


# ---------------------------------------------------------------------------
# classify_orf_type
# ---------------------------------------------------------------------------

class TestClassifyOrfType:
    """
    Canonical CDS: cstart=100, cstop=400 (in-frame = start%3 == cstart%3 = 1)
    """
    CSTART = 100
    CSTOP  = 400

    def _cls(self, start, stop, out_of_frame, name='TX1'):
        return classify_orf_type(start, stop, self.CSTART, self.CSTOP,
                                 out_of_frame, name, novel_prefix='MSTRG')

    # --- noncoding / novel ---
    def test_noncoding(self):
        assert classify_orf_type(10, 50, -1, 400, True, 'ENST1', 'MSTRG') == 'noncoding'

    def test_novel(self):
        assert classify_orf_type(10, 50, -1, 400, True, 'MSTRG.1', 'MSTRG') == 'novel'

    # --- out-of-frame upstream ---
    def test_uorf(self):
        assert self._cls(10, 80, out_of_frame=True) == 'uORF'

    def test_overlap_uorf_outframe(self):
        assert self._cls(10, 200, out_of_frame=True) == 'overlap.uORF'

    def test_polycistronic_upstream_outframe(self):
        assert self._cls(10, 450, out_of_frame=True) == 'polycistronic'

    # --- out-of-frame downstream ---
    def test_internal_outframe(self):
        assert self._cls(150, 390, out_of_frame=True) == 'internal'

    def test_external_outframe(self):
        assert self._cls(150, 420, out_of_frame=True) == 'external'

    def test_polycistronic_downstream_outframe(self):
        assert self._cls(410, 500, out_of_frame=True) == 'polycistronic'

    # --- in-frame ---
    def test_canonical(self):
        assert self._cls(100, 400, out_of_frame=False) == 'canonical'

    def test_internal_inframe_short(self):
        # start == cstart but stop != cstop
        assert self._cls(100, 300, out_of_frame=False) == 'internal'

    def test_extension(self):
        assert self._cls(70, 400, out_of_frame=False) == 'extension'

    def test_truncation(self):
        assert self._cls(130, 400, out_of_frame=False) == 'truncation'

    def test_uorf_inframe(self):
        assert self._cls(10, 80, out_of_frame=False) == 'uORF'

    def test_overlap_uorf_inframe(self):
        assert self._cls(10, 150, out_of_frame=False) == 'overlap.uORF'

    def test_internal_inframe_downstream(self):
        assert self._cls(150, 350, out_of_frame=False) == 'internal'

    def test_polycistronic_inframe_downstream(self):
        assert self._cls(150, 450, out_of_frame=False) == 'polycistronic'


# ---------------------------------------------------------------------------
# find_local_maxima
# ---------------------------------------------------------------------------

class TestFindLocalMaxima:

    def _make_sc(self, scores_by_tid: dict) -> pd.DataFrame:
        rows = []
        for tid, scores in scores_by_tid.items():
            for i, s in enumerate(scores):
                rows.append({'transcript ID': tid, 'start': i * 10, 'cosine similarity': s})
        df = pd.DataFrame(rows)
        df.index = df['transcript ID'] + '_' + df['start'].astype(str)
        df.index.name = 'TIS ID'
        return df

    def test_single_peak(self):
        sc = self._make_sc({'TX1': [0.1, 0.9, 0.3]})
        lmax = find_local_maxima(sc)
        assert 'TX1_10' in lmax  # peak at index 1

    def test_no_local_max(self):
        # Monotonically decreasing: no local max (sentinel -1 prepended,
        # so only the first element could qualify if it's > -1 — it is)
        sc = self._make_sc({'TX1': [0.9, 0.5, 0.1]})
        lmax = find_local_maxima(sc)
        assert 'TX1_0' in lmax

    def test_multiple_transcripts(self):
        # TX1: peak at index 1 (0.8 > 0.1 and 0.8 > 0.2)
        # TX2: peak at index 1 (0.9 > 0.3 and 0.9 > 0.4); endpoint 0.9 is NOT
        #      detected by argrelmax (endpoints are excluded by scipy convention)
        sc = self._make_sc({
            'TX1': [0.1, 0.8, 0.2],
            'TX2': [0.3, 0.9, 0.4],
        })
        lmax = find_local_maxima(sc)
        assert 'TX1_10' in lmax
        assert 'TX2_10' in lmax

    def test_returns_list(self):
        sc = self._make_sc({'TX1': [0.5, 0.9, 0.4]})
        assert isinstance(find_local_maxima(sc), list)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:

    BASE = [
        '--fasta', 'tx.fa', '--cosine', 'cos.csv',
        '--position', 'pos.csv', '--symbol', 'sym.csv',
        '--ref', 'ref.csv', '--name', 'test'
    ]

    def test_required_args(self):
        args = parse_args(self.BASE)
        assert args.name == 'test'
        assert args.fasta == 'tx.fa'

    def test_defaults(self):
        args = parse_args(self.BASE)
        assert args.min_length_coverage == 10
        assert args.novel_prefix == 'MSTRG'

    def test_custom_values(self):
        args = parse_args(self.BASE + ['--min-length-coverage', '5', '--novel-prefix', 'STRG'])
        assert args.min_length_coverage == 5
        assert args.novel_prefix == 'STRG'

    def test_threshold_default_is_none(self):
        args = parse_args(self.BASE)
        assert args.threshold is None

    def test_threshold_custom(self):
        args = parse_args(self.BASE + ['--threshold', '0.42'])
        assert args.threshold == pytest.approx(0.42)

    def test_missing_required(self):
        with pytest.raises(SystemExit):
            parse_args(['--fasta', 'tx.fa'])
