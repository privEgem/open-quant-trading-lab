"""Extraction regressions and independently constructed semantic checks."""
from importlib import import_module
from pathlib import Path
import pkgutil
import unittest

import trading_lab
from trading_lab.demo import run_demo
from trading_lab.research_audit.evidence_identity import canonical_sha256, material_code_fingerprint


class ReleaseContractTests(unittest.TestCase):
    def test_all_toolkit_modules_import_without_network_or_runtime_files(self):
        for module in pkgutil.walk_packages(trading_lab.__path__, trading_lab.__name__ + "."):
            import_module(module.name)
        self.assertFalse((Path(__file__).parents[1] / "state").exists())

    def test_full_synthetic_backtest_closes_positions_and_reconciles_cash(self):
        result = run_demo()
        self.assertEqual(result["bars"], 1950)
        self.assertGreater(result["trades"], 0)
        self.assertEqual(result["open_position_at_end"], 0)
        self.assertAlmostEqual(result["cash_change"], result["trade_pnl_less_entry_commission"], places=5)
        self.assertLessEqual(result["max_drawdown_pct"], 0)

    def test_canonical_identity_rejects_nonfinite_and_duplicate_roles(self):
        self.assertEqual(canonical_sha256({"a": 1, "b": 2}), canonical_sha256({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            canonical_sha256({"a": float("nan")})
        component = {"role": "engine", "path": "engine.py", "git_blob_oid": "a" * 40, "git_blob_sha256": "b" * 64}
        with self.assertRaises(ValueError):
            material_code_fingerprint([component, component])
