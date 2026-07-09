"""Known-good-output test against the compiled COSP Fortran radar path.

Requires the reference binary built by ``tools/kgo/build.sh`` (gfortran + a
COSPv2.0 checkout); skipped otherwise. See docs/validation.md.
"""

import pathlib
import sys
import unittest

_KGO_DIR = pathlib.Path(__file__).resolve().parent.parent / "tools" / "kgo"


@unittest.skipUnless((_KGO_DIR / "drv_radar").is_file(),
                     "KGO binary not built (tools/kgo/build.sh)")
class KgoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_KGO_DIR))
        import compare_kgo
        cls.kgo = compare_kgo

    def test_exact_regime(self):
        # Temperatures on the LUT bin grid: the lazy Fortran table fill and
        # jcosp's offline table coincide; agreement to the numerical floor.
        case = self.kgo.make_case(snap_temperature=True)
        report = self.kgo.compare(self.kgo.run_fortran(case),
                                  self.kgo.run_jcosp(case),
                                  exact=True, verbose=False)
        self.assertTrue(report["prec_frac_identical"])

    def test_realistic_regime(self):
        # Arbitrary temperatures: bounded by the temperature-bin evaluation
        # difference (documented in docs/translation_notes.md).
        case = self.kgo.make_case(snap_temperature=False)
        self.kgo.compare(self.kgo.run_fortran(case),
                         self.kgo.run_jcosp(case),
                         exact=False, verbose=False)


if __name__ == "__main__":
    unittest.main()
