"""Independent fictional replacement for excluded broker-incident fixture."""
from datetime import datetime, timezone
from decimal import Decimal
import unittest

from trading_lab.risk_and_safety.reconciliation import (
    BrokerQueryState, BrokerSnapshot, PositionView, ReconciliationStatus,
    Severity, reconcile_snapshot,
)


class SyntheticReconciliationTests(unittest.TestCase):
    def test_unprotected_short_never_allows_entry(self):
        snapshot = BrokerSnapshot(
            observed_at=datetime(2025, 1, 6, 15, tzinfo=timezone.utc),
            position_query_state=BrokerQueryState.OK,
            order_query_state=BrokerQueryState.OK,
            positions_complete=True,
            orders_complete=True,
            positions=(PositionView("SYNTH", Decimal("-7")),),
            orders=(),
        )
        result = reconcile_snapshot(snapshot, strategy_symbol="SYNTH")
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_UNPROTECTED)
        self.assertEqual(result.severity, Severity.CRITICAL)
        self.assertFalse(result.entry_allowed)
        self.assertEqual(result.coverage_summaries[0].required_quantity, Decimal("7"))
