import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from trading_lab.risk_and_safety.reconciliation import (
    MAX_CONSECUTIVE_UNKNOWN_SNAPSHOTS,
    MAX_ORDER_TIMESTAMP_FUTURE_SKEW_SECONDS,
    MAX_UNKNOWN_ELAPSED_SECONDS,
    PENDING_CLOSE_STALE_AFTER_SECONDS,
    REQUIRED_HEALTHY_RECOVERY_SNAPSHOTS,
    BrokerQueryState,
    BrokerNormalizationError,
    BrokerSnapshot,
    ExpectedCloseContext,
    ExpectedEntryContext,
    HoldingPolicy,
    PositionView,
    ReconciliationStatus,
    SafetyState,
    SafetyIncidentState,
    Severity,
    apply_safety_state_transition,
    can_submit_entry,
    classify_pending_entry,
    flatten_order_tree,
    normalize_order,
    normalize_position,
    reconcile_snapshot,
)


UTC = timezone.utc
T0 = datetime(2026, 7, 14, 14, 0, tzinfo=UTC)
DEFAULT_ORDER_PRICE = object()


def order(
    order_id,
    *,
    symbol="SPY",
    side="buy",
    qty="13",
    filled="0",
    status="new",
    order_type="market",
    order_class="simple",
    parent_id=None,
    submitted_at=T0,
    client_order_id=None,
    time_in_force="day",
    limit_price=DEFAULT_ORDER_PRICE,
    stop_price=DEFAULT_ORDER_PRICE,
    trail_price=DEFAULT_ORDER_PRICE,
    trail_percent=DEFAULT_ORDER_PRICE,
):
    raw = {
        "id": order_id,
        "client_order_id": client_order_id or f"client-{order_id}",
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "filled_qty": filled,
        "status": status,
        "type": order_type,
        "order_class": order_class,
        "time_in_force": time_in_force,
        "submitted_at": submitted_at,
        "updated_at": submitted_at,
    }
    if limit_price is DEFAULT_ORDER_PRICE:
        limit_price = "450" if order_type in {"limit", "stop_limit"} else None
    if stop_price is DEFAULT_ORDER_PRICE:
        stop_price = "400" if order_type in {"stop", "stop_limit"} else None
    if trail_price is DEFAULT_ORDER_PRICE:
        trail_price = None
    if trail_percent is DEFAULT_ORDER_PRICE:
        trail_percent = "1" if order_type == "trailing_stop" else None
    for name, value in (
        ("limit_price", limit_price),
        ("stop_price", stop_price),
        ("trail_price", trail_price),
        ("trail_percent", trail_percent),
    ):
        if value is not None:
            raw[name] = value
    if parent_id:
        raw["parent_order_id"] = parent_id
    return normalize_order(raw)


def snapshot(
    *,
    positions=(),
    orders=(),
    observed_at=T0,
    position_state=BrokerQueryState.OK,
    order_state=BrokerQueryState.OK,
    positions_complete=True,
    orders_complete=True,
    positions_at_cap=False,
    orders_at_cap=False,
):
    return BrokerSnapshot(
        positions=tuple(positions),
        orders=tuple(orders),
        position_query_state=position_state,
        order_query_state=order_state,
        positions_complete=positions_complete,
        orders_complete=orders_complete,
        positions_at_known_cap=positions_at_cap,
        orders_at_known_cap=orders_at_cap,
        observed_at=observed_at,
    )


def expected_entry(
    order_id="entry-1",
    *,
    client_order_id=None,
    broker_order_id=None,
    symbol="SPY",
    side="buy",
    quantity=Decimal("13"),
    order_class="simple",
    order_type=None,
    time_in_force=None,
    stop_price=None,
    take_profit_limit_price=None,
    request_fingerprint=None,
):
    return ExpectedEntryContext(
        client_order_id=client_order_id or f"client-{order_id}",
        broker_order_id=order_id if broker_order_id is None else broker_order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_class=order_class,
        order_type=order_type,
        time_in_force=time_in_force,
        stop_price=stop_price,
        take_profit_limit_price=take_profit_limit_price,
        request_fingerprint=request_fingerprint,
    )


def expected_close(
    order_id="close-1",
    *,
    client_order_id=None,
    broker_order_id=None,
    symbol="SPY",
    side="sell",
    remaining_quantity=Decimal("13"),
    order_type="market",
):
    return ExpectedCloseContext(
        client_order_id=client_order_id or f"client-{order_id}",
        broker_order_id=order_id if broker_order_id is None else broker_order_id,
        symbol=symbol,
        side=side,
        remaining_quantity=remaining_quantity,
        order_type=order_type,
    )


def result_for(*, expected=None, expected_close_context=None, **kwargs):
    return reconcile_snapshot(
        snapshot(**kwargs),
        expected_entry=expected,
        expected_close=expected_close_context,
    )


def stop(
    order_id,
    *,
    position_side="long",
    qty="13",
    side=None,
    status="new",
    parent_id="parent",
    order_type="stop",
    stop_price=DEFAULT_ORDER_PRICE,
    limit_price=DEFAULT_ORDER_PRICE,
    trail_price=DEFAULT_ORDER_PRICE,
    trail_percent=DEFAULT_ORDER_PRICE,
):
    protective_side = side or ("sell" if position_side == "long" else "buy")
    return order(
        order_id,
        side=protective_side,
        qty=qty,
        status=status,
        order_type=order_type,
        order_class="bracket",
        parent_id=parent_id,
        stop_price=stop_price,
        limit_price=limit_price,
        trail_price=trail_price,
        trail_percent=trail_percent,
    )


def healthy_flat(at=T0):
    return reconcile_snapshot(snapshot(observed_at=at))


def unknown(at=T0, *, state=BrokerQueryState.UNKNOWN):
    return reconcile_snapshot(snapshot(observed_at=at, position_state=state))


class BrokerReconciliationTests(unittest.TestCase):
    def test_flat_no_orders_is_healthy_but_policy_unresolved_blocks_entry(self):
        result = healthy_flat()
        self.assertEqual(result.status, ReconciliationStatus.HEALTHY_FLAT)
        self.assertEqual(result.severity, Severity.INFO)
        self.assertFalse(result.entry_allowed)
        self.assertFalse(result.incident_required)
        resolved = reconcile_snapshot(snapshot(), holding_policy=HoldingPolicy.INTRADAY_FLAT)
        self.assertFalse(resolved.entry_allowed)

    def test_flat_arbitrary_entry_without_expected_context_is_operator_visible_mismatch(self):
        result = result_for(orders=(order("entry-1", side="buy"),))
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("EXPECTED_ENTRY_CONTEXT_MISSING", result.issue_codes)
        self.assertTrue(result.incident_required)
        self.assertTrue(result.operator_action_required)
        self.assertFalse(result.entry_allowed)

    def test_flat_exact_expected_entry_is_controlled_not_incident(self):
        result = result_for(
            orders=(order("entry-1", side="buy"),),
            expected=expected_entry(),
        )
        self.assertEqual(result.status, ReconciliationStatus.CONTROLLED_ENTRY_PENDING)
        self.assertFalse(result.entry_allowed)
        self.assertFalse(result.incident_required)
        self.assertTrue(result.auto_recoverable)

    def test_pending_bracket_matches_full_persisted_request_shape(self):
        root = order("entry-1", order_class="bracket")
        stop_child = order(
            "stop-1", side="sell", order_type="stop", order_class="bracket",
            parent_id="entry-1", stop_price="399.25", status="held",
        )
        take_profit = order(
            "tp-1", side="sell", order_type="limit", order_class="bracket",
            parent_id="entry-1", limit_price="451.75", status="held",
        )
        expected = expected_entry(
            order_class="bracket",
            order_type="market",
            time_in_force="day",
            stop_price=Decimal("399.25"),
            take_profit_limit_price=Decimal("451.75"),
            request_fingerprint="sha256:" + "a" * 64,
        )
        exact = result_for(orders=(root, stop_child, take_profit), expected=expected)
        self.assertEqual(exact.status, ReconciliationStatus.CONTROLLED_ENTRY_PENDING)

        wrong_stop = replace(stop_child, stop_price=Decimal("398"))
        mismatch = result_for(orders=(root, wrong_stop, take_profit), expected=expected)
        self.assertEqual(mismatch.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("EXPECTED_ENTRY_STOP_PRICE_MISMATCH", mismatch.issue_codes)

    def test_pending_entry_classifier_requires_complete_flat_snapshot_evidence(self):
        candidate = order("entry-1")
        classification = classify_pending_entry(
            (candidate,), expected_entry=expected_entry()
        )
        self.assertFalse(classification.matched)
        self.assertTrue(classification.unsafe)
        self.assertIn("PENDING_ENTRY_SNAPSHOT_UNVERIFIED", classification.issue_codes)

    def test_flat_duplicate_pending_entry_groups_escalate(self):
        result = result_for(
            orders=(order("entry-1"), order("entry-2")), expected=expected_entry()
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertTrue(result.incident_required)
        self.assertIn("DUPLICATE_PENDING_ENTRY_GROUPS", result.issue_codes)

    def test_unresolved_submission_ambiguity_is_transient_until_recovery_exhausts(self):
        result = reconcile_snapshot(
            snapshot(), transition_context={"submission_ambiguous_unresolved": True}
        )
        self.assertEqual(result.status, ReconciliationStatus.BLOCKED_UNKNOWN_TRANSIENT)
        self.assertFalse(result.incident_required)
        self.assertTrue(result.auto_recoverable)
        self.assertIn("SUBMISSION_AMBIGUITY_UNRESOLVED", result.issue_codes)

    def test_stale_pending_entry_is_only_evidence(self):
        result = result_for(
            orders=(order("entry-old", submitted_at=T0),),
            observed_at=T0 + timedelta(seconds=301),
            expected=expected_entry("entry-old"),
        )
        self.assertEqual(result.status, ReconciliationStatus.CONTROLLED_ENTRY_PENDING)
        self.assertIn("PENDING_ENTRY_STALE_EVIDENCE", result.issue_codes)
        self.assertFalse(result.incident_required)

    def test_flat_partially_filled_root_is_never_controlled_pending(self):
        result = result_for(
            orders=(
                order(
                    "entry-1",
                    status="partially_filled",
                    filled="1",
                ),
            ),
            expected=expected_entry(),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("FLAT_WITH_PARTIALLY_FILLED_ENTRY", result.issue_codes)
        self.assertFalse(result.auto_recoverable)

    def test_flat_new_root_with_positive_filled_quantity_is_never_controlled_pending(self):
        result = result_for(
            orders=(order("entry-1", status="new", filled="1"),),
            expected=expected_entry(),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("FLAT_WITH_PARTIALLY_FILLED_ENTRY", result.issue_codes)

    def test_expected_entry_provenance_mismatches_are_explicit(self):
        cases = (
            (
                expected_entry(broker_order_id="other-broker-id"),
                "EXPECTED_ENTRY_ID_MISMATCH",
            ),
            (
                expected_entry(client_order_id="other-client-id"),
                "EXPECTED_ENTRY_CLIENT_ID_MISMATCH",
            ),
            (expected_entry(symbol="QQQ"), "EXPECTED_ENTRY_SYMBOL_MISMATCH"),
            (expected_entry(side="sell"), "EXPECTED_ENTRY_SIDE_MISMATCH"),
            (
                expected_entry(quantity=Decimal("13.0001")),
                "EXPECTED_ENTRY_QUANTITY_MISMATCH",
            ),
            (
                expected_entry(order_class="bracket"),
                "EXPECTED_ENTRY_ORDER_CLASS_MISMATCH",
            ),
        )
        for expected, issue_code in cases:
            with self.subTest(issue_code=issue_code):
                result = result_for(orders=(order("entry-1"),), expected=expected)
                self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
                self.assertIn(issue_code, result.issue_codes)

    def test_expected_entry_quantity_requires_decimal(self):
        with self.assertRaisesRegex(BrokerNormalizationError, "exact Decimal"):
            ExpectedEntryContext(
                client_order_id="client-entry-1",
                broker_order_id="entry-1",
                symbol="SPY",
                side="buy",
                quantity="13",
                order_class="simple",
            )

    def test_pending_bracket_terminal_stop_is_invalid(self):
        for status in ("canceled", "expired"):
            with self.subTest(status=status):
                root = order("entry-1", order_class="bracket")
                stop_leg = order(
                    "stop-1",
                    side="sell",
                    status=status,
                    order_type="stop",
                    order_class="bracket",
                    parent_id="entry-1",
                )
                take_profit = order(
                    "take-profit-1",
                    side="sell",
                    status="held",
                    order_type="limit",
                    order_class="bracket",
                    parent_id="entry-1",
                )
                result = result_for(
                    orders=(root, stop_leg, take_profit),
                    expected=expected_entry(order_class="bracket"),
                )
                self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
                self.assertIn("PENDING_BRACKET_STOP_INVALID", result.issue_codes)

    def test_pending_bracket_take_profit_and_child_shape_must_be_coherent(self):
        cases = (
            (
                order(
                    "take-profit-1",
                    side="sell",
                    status="canceled",
                    order_type="limit",
                    order_class="bracket",
                    parent_id="entry-1",
                ),
                "PENDING_BRACKET_TAKE_PROFIT_INVALID",
            ),
            (
                order(
                    "take-profit-1",
                    side="buy",
                    status="held",
                    order_type="limit",
                    order_class="bracket",
                    parent_id="entry-1",
                ),
                "PENDING_BRACKET_TAKE_PROFIT_INVALID",
            ),
            (
                order(
                    "take-profit-1",
                    side="sell",
                    qty="12.999",
                    status="held",
                    order_type="limit",
                    order_class="bracket",
                    parent_id="entry-1",
                ),
                "PENDING_BRACKET_TAKE_PROFIT_INVALID",
            ),
        )
        for take_profit, issue_code in cases:
            with self.subTest(status=take_profit.status, side=take_profit.side):
                root = order("entry-1", order_class="bracket")
                stop_leg = order(
                    "stop-1",
                    side="sell",
                    status="held",
                    order_type="stop",
                    order_class="bracket",
                    parent_id="entry-1",
                )
                result = result_for(
                    orders=(root, stop_leg, take_profit),
                    expected=expected_entry(order_class="bracket"),
                )
                self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
                self.assertIn("PENDING_BRACKET_STRUCTURE_INVALID", result.issue_codes)
                self.assertIn(issue_code, result.issue_codes)

    def test_pending_bracket_duplicate_stops_are_invalid(self):
        root = order("entry-1", order_class="bracket")
        children = (
            order(
                "stop-1",
                side="sell",
                status="held",
                order_type="stop",
                order_class="bracket",
                parent_id="entry-1",
            ),
            order(
                "stop-2",
                side="sell",
                status="inactive",
                order_type="stop_limit",
                order_class="bracket",
                parent_id="entry-1",
            ),
            order(
                "take-profit-1",
                side="sell",
                status="held",
                order_type="limit",
                order_class="bracket",
                parent_id="entry-1",
            ),
        )
        result = result_for(
            orders=(root, *children), expected=expected_entry(order_class="bracket")
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_BRACKET_STRUCTURE_INVALID", result.issue_codes)
        self.assertIn("PENDING_BRACKET_STOP_INVALID", result.issue_codes)

    def test_pending_bracket_held_and_inactive_children_are_coherent_preactivation(self):
        root = order("entry-1", order_class="bracket")
        stop_leg = order(
            "stop-1",
            side="sell",
            status="held",
            order_type="stop",
            order_class="bracket",
            parent_id="entry-1",
        )
        take_profit = order(
            "take-profit-1",
            side="sell",
            status="inactive",
            order_type="limit",
            order_class="bracket",
            parent_id="entry-1",
        )
        result = result_for(
            orders=(root, stop_leg, take_profit),
            expected=expected_entry(order_class="bracket"),
        )
        self.assertEqual(result.status, ReconciliationStatus.CONTROLLED_ENTRY_PENDING)
        self.assertFalse(result.incident_required)

    def test_pending_bracket_stop_without_stop_price_is_rejected(self):
        root = order("entry-1", order_class="bracket")
        stop_leg = stop(
            "stop-1", status="held", parent_id="entry-1", stop_price=None
        )
        take_profit = order(
            "take-profit-1",
            side="sell",
            status="inactive",
            order_type="limit",
            order_class="bracket",
            parent_id="entry-1",
        )
        result = result_for(
            orders=(root, stop_leg, take_profit),
            expected=expected_entry(order_class="bracket"),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_BRACKET_STOP_INVALID", result.issue_codes)

    def test_pending_bracket_take_profit_without_limit_price_is_rejected(self):
        root = order("entry-1", order_class="bracket")
        stop_leg = stop("stop-1", status="held", parent_id="entry-1")
        take_profit = order(
            "take-profit-1",
            side="sell",
            status="inactive",
            order_type="limit",
            order_class="bracket",
            parent_id="entry-1",
            limit_price=None,
        )
        result = result_for(
            orders=(root, stop_leg, take_profit),
            expected=expected_entry(order_class="bracket"),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_BRACKET_TAKE_PROFIT_INVALID", result.issue_codes)

    def test_exact_matching_stop_protects_long_and_short(self):
        cases = (
            (PositionView("SPY", Decimal("13")), stop("stop-long", position_side="long")),
            (PositionView("SPY", Decimal("-13")), stop("stop-short", position_side="short")),
        )
        for position, protective_order in cases:
            with self.subTest(position=position.quantity):
                result = result_for(positions=(position,), orders=(protective_order,))
                self.assertEqual(result.status, ReconciliationStatus.HEALTHY_PROTECTED)
                self.assertFalse(result.entry_allowed)
                summary = result.coverage_summaries[0]
                self.assertEqual(summary.valid_active_stop_quantity, Decimal("13"))
                self.assertEqual(summary.uncovered_quantity, 0)
                self.assertEqual(summary.overcovered_quantity, 0)

    def test_active_stop_requires_a_finite_positive_stop_price(self):
        cases = ((None, "missing"), ("0", "zero"), ("-1", "negative"))
        for stop_price, label in cases:
            with self.subTest(price=label):
                result = result_for(
                    positions=(PositionView("SPY", Decimal("13")),),
                    orders=(stop(f"invalid-{label}", stop_price=stop_price),),
                )
                self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
                summary = result.coverage_summaries[0]
                self.assertEqual(summary.valid_active_stop_quantity, Decimal("0"))
                self.assertIn("PROTECTION_STOP_PRICE_INVALID", summary.issue_codes)

        valid = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(stop("valid-positive", stop_price="400.01"),),
        )
        self.assertEqual(valid.status, ReconciliationStatus.HEALTHY_PROTECTED)
        self.assertEqual(
            valid.coverage_summaries[0].valid_active_stop_quantity, Decimal("13")
        )

    def test_valid_full_stop_plus_active_stop_missing_price_is_mismatch(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(
                stop("valid-full", stop_price="400"),
                stop("invalid-missing", stop_price=None),
            ),
        )

        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertEqual(result.severity, Severity.CRITICAL)
        self.assertTrue(result.incident_required)
        self.assertTrue(result.operator_action_required)
        self.assertIn("PROTECTION_STOP_PRICE_INVALID", result.issue_codes)

    def test_valid_full_stop_plus_active_zero_price_stop_is_mismatch(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(
                stop("valid-full", stop_price="400"),
                stop("invalid-zero", stop_price="0"),
            ),
        )

        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PROTECTION_STOP_PRICE_INVALID", result.issue_codes)

    def test_valid_full_stop_alone_remains_healthy(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(stop("valid-full", stop_price="400"),),
        )

        self.assertEqual(result.status, ReconciliationStatus.HEALTHY_PROTECTED)
        self.assertNotIn("PROTECTION_STOP_PRICE_INVALID", result.issue_codes)

    def test_valid_trailing_stop_with_positive_trail_value_is_healthy(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(
                stop(
                    "valid-trailing",
                    order_type="trailing_stop",
                    stop_price=None,
                    trail_price=None,
                    trail_percent="1.5",
                ),
            ),
        )

        self.assertEqual(result.status, ReconciliationStatus.HEALTHY_PROTECTED)
        self.assertNotIn("PROTECTION_STOP_PRICE_INVALID", result.issue_codes)

    def test_canceled_malformed_stop_does_not_create_active_price_incident(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(
                stop("valid-full", stop_price="400"),
                stop("historical-canceled", status="canceled", stop_price=None),
            ),
        )

        self.assertEqual(result.status, ReconciliationStatus.HEALTHY_PROTECTED)
        self.assertNotIn("PROTECTION_STOP_PRICE_INVALID", result.issue_codes)

    def test_active_stop_limit_requires_both_positive_prices(self):
        cases = (
            (None, "401", False),
            ("400", None, False),
            ("0", "401", False),
            ("400", "-1", False),
            ("400", "401", True),
        )
        for stop_price, limit_price, protects in cases:
            with self.subTest(stop_price=stop_price, limit_price=limit_price):
                result = result_for(
                    positions=(PositionView("SPY", Decimal("13")),),
                    orders=(
                        stop(
                            "stop-limit",
                            order_type="stop_limit",
                            stop_price=stop_price,
                            limit_price=limit_price,
                        ),
                    ),
                )
                summary = result.coverage_summaries[0]
                self.assertEqual(
                    summary.valid_active_stop_quantity,
                    Decimal("13") if protects else Decimal("0"),
                )
                if protects:
                    self.assertEqual(result.status, ReconciliationStatus.HEALTHY_PROTECTED)
                else:
                    self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
                    self.assertIn("PROTECTION_STOP_PRICE_INVALID", summary.issue_codes)

    def test_short_without_orders_is_critical_unprotected(self):
        result = result_for(positions=(PositionView("SPY", Decimal("-13")),))
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_UNPROTECTED)
        self.assertEqual(result.severity, Severity.CRITICAL)
        self.assertFalse(result.entry_allowed)
        self.assertEqual(result.coverage_summaries[0].required_protective_side, "BUY")


    def test_undercoverage_and_overcoverage_are_exact_and_critical(self):
        cases = (
            ("8", ReconciliationStatus.INCIDENT_UNDERCOVERED, Decimal("5"), Decimal("0")),
            ("20", ReconciliationStatus.INCIDENT_OVERCOVERED, Decimal("0"), Decimal("7")),
        )
        for qty, expected, uncovered, overcovered in cases:
            with self.subTest(qty=qty):
                result = result_for(
                    positions=(PositionView("SPY", Decimal("13")),),
                    orders=(stop("coverage-stop", qty=qty),),
                )
                self.assertEqual(result.status, expected)
                self.assertEqual(result.severity, Severity.CRITICAL)
                self.assertEqual(result.coverage_summaries[0].uncovered_quantity, uncovered)
                self.assertEqual(result.coverage_summaries[0].overcovered_quantity, overcovered)

    def test_wrong_side_stop_has_zero_effective_coverage(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(stop("wrong-side", side="buy"),),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_WRONG_SIDE)
        self.assertEqual(result.coverage_summaries[0].valid_active_stop_quantity, 0)
        self.assertIn("WRONG_SIDE_PROTECTION", result.issue_codes)

    def test_take_profit_is_not_loss_protection(self):
        take_profit = order(
            "tp", side="sell", order_type="limit", order_class="bracket", parent_id="parent"
        )
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),), orders=(take_profit,)
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_UNPROTECTED)
        self.assertEqual(result.coverage_summaries[0].valid_active_stop_quantity, 0)

    def test_oco_take_profit_and_stop_are_not_added_as_double_coverage(self):
        take_profit = order(
            "tp", side="sell", order_type="limit", order_class="bracket", parent_id="parent"
        )
        protective = stop("sl", position_side="long", parent_id="parent")
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(take_profit, protective),
        )
        self.assertEqual(result.status, ReconciliationStatus.HEALTHY_PROTECTED)
        self.assertEqual(result.coverage_summaries[0].valid_active_stop_quantity, Decimal("13"))

    def test_partially_filled_parent_without_active_child_is_unprotected(self):
        parent = order(
            "partial-parent",
            side="sell",
            qty="13",
            filled="5",
            status="partially_filled",
            order_class="bracket",
        )
        result = result_for(
            positions=(PositionView("SPY", Decimal("-5")),), orders=(parent,)
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_UNPROTECTED)
        self.assertEqual(result.coverage_summaries[0].uncovered_quantity, Decimal("5"))

    def test_partially_filled_parent_with_exact_stop_is_protected_but_blocked(self):
        parent = order(
            "partial-parent",
            side="sell",
            qty="13",
            filled="5",
            status="partially_filled",
            order_class="bracket",
        )
        child = stop("synthetic-stop", position_side="short", qty="5", parent_id="partial-parent")
        result = result_for(
            positions=(PositionView("SPY", Decimal("-5")),), orders=(parent, child)
        )
        self.assertEqual(result.status, ReconciliationStatus.HEALTHY_PROTECTED)
        self.assertFalse(result.entry_allowed)

    def test_close_without_expected_context_is_incident_mismatch(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(order("close-1", side="sell"),),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("EXPECTED_CLOSE_CONTEXT_MISSING", result.issue_codes)

    def test_seven_day_old_gtc_limit_close_is_never_controlled(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(
                order(
                    "close-1",
                    side="sell",
                    order_type="limit",
                    limit_price="9999",
                    time_in_force="gtc",
                    submitted_at=T0 - timedelta(days=7),
                ),
            ),
            expected_close_context=expected_close(order_type="limit"),
            observed_at=T0,
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("EXPECTED_CLOSE_TYPE_MISMATCH", result.issue_codes)
        self.assertIn("PENDING_CLOSE_STALE", result.issue_codes)

    def test_fresh_expected_limit_close_is_never_controlled(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(order("close-1", side="sell", order_type="limit"),),
            expected_close_context=expected_close(order_type="limit"),
            observed_at=T0 + timedelta(seconds=1),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("EXPECTED_CLOSE_TYPE_MISMATCH", result.issue_codes)

    def test_held_market_close_is_never_controlled(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(order("close-1", side="sell", status="held"),),
            expected_close_context=expected_close(),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_CLOSE_STATUS_INVALID", result.issue_codes)

    def test_market_close_missing_submitted_at_is_never_controlled(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(order("close-1", side="sell", submitted_at=None),),
            expected_close_context=expected_close(),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_CLOSE_TIMESTAMP_MISSING", result.issue_codes)

    def test_stale_expected_market_close_is_incident_mismatch(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(order("close-1", side="sell", submitted_at=T0),),
            expected_close_context=expected_close(),
            observed_at=T0 + timedelta(seconds=PENDING_CLOSE_STALE_AFTER_SECONDS + 1),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_CLOSE_STALE", result.issue_codes)
        self.assertTrue(result.incident_required)

    def test_expected_close_six_seconds_in_future_is_incident_mismatch(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(
                order(
                    "close-1",
                    side="sell",
                    submitted_at=T0
                    + timedelta(seconds=MAX_ORDER_TIMESTAMP_FUTURE_SKEW_SECONDS + 1),
                ),
            ),
            expected_close_context=expected_close(),
            observed_at=T0,
        )

        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_CLOSE_TIMESTAMP_FUTURE", result.issue_codes)
        self.assertTrue(result.incident_required)
        self.assertTrue(result.operator_action_required)

    def test_expected_close_at_future_skew_limit_is_controlled(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(
                order(
                    "close-1",
                    side="sell",
                    submitted_at=T0
                    + timedelta(seconds=MAX_ORDER_TIMESTAMP_FUTURE_SKEW_SECONDS),
                ),
            ),
            expected_close_context=expected_close(),
            observed_at=T0,
        )

        self.assertEqual(result.status, ReconciliationStatus.CONTROLLED_CLOSING)

    def test_expected_close_at_observation_time_is_controlled(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(order("close-1", side="sell", submitted_at=T0),),
            expected_close_context=expected_close(),
            observed_at=T0,
        )

        self.assertEqual(result.status, ReconciliationStatus.CONTROLLED_CLOSING)

    def test_expected_close_at_stale_age_limit_is_controlled(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(order("close-1", side="sell", submitted_at=T0),),
            expected_close_context=expected_close(),
            observed_at=T0 + timedelta(seconds=PENDING_CLOSE_STALE_AFTER_SECONDS),
        )

        self.assertEqual(result.status, ReconciliationStatus.CONTROLLED_CLOSING)

    def test_fresh_expected_market_close_with_exact_provenance_is_controlled(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(order("close-1", side="sell", submitted_at=T0),),
            expected_close_context=expected_close(),
            observed_at=T0 + timedelta(seconds=30),
        )
        self.assertEqual(result.status, ReconciliationStatus.CONTROLLED_CLOSING)
        self.assertFalse(result.incident_required)
        self.assertFalse(result.entry_allowed)

    def test_expected_close_client_order_id_mismatch_is_incident(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(order("close-1", side="sell"),),
            expected_close_context=expected_close(client_order_id="different-client"),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("EXPECTED_CLOSE_CLIENT_ID_MISMATCH", result.issue_codes)

    def test_expected_close_broker_order_id_mismatch_is_incident(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(order("close-1", side="sell"),),
            expected_close_context=expected_close(broker_order_id="different-broker-id"),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("EXPECTED_CLOSE_ID_MISMATCH", result.issue_codes)

    def test_pending_close_with_another_active_order_is_not_coherent(self):
        closing = order("close-1", side="sell")
        protective = stop("still-active", position_side="long")
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(closing, protective),
            expected_close_context=expected_close(),
        )
        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_CLOSE_WITH_OTHER_ACTIVE_ORDERS", result.issue_codes)

    def test_partially_filled_close_remaining_must_match_position(self):
        coherent = result_for(
            positions=(PositionView("SPY", Decimal("8")),),
            orders=(
                order(
                    "close-ok",
                    side="sell",
                    qty="13",
                    filled="5",
                    status="partially_filled",
                ),
            ),
            expected_close_context=expected_close(
                "close-ok", remaining_quantity=Decimal("8")
            ),
        )
        self.assertEqual(coherent.status, ReconciliationStatus.CONTROLLED_CLOSING)
        mismatch = result_for(
            positions=(PositionView("SPY", Decimal("7")),),
            orders=(
                order(
                    "close-bad",
                    side="sell",
                    qty="13",
                    filled="5",
                    status="partially_filled",
                ),
            ),
            expected_close_context=expected_close(
                "close-bad", remaining_quantity=Decimal("7")
            ),
        )
        self.assertEqual(mismatch.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("EXPECTED_CLOSE_QUANTITY_MISMATCH", mismatch.issue_codes)

    def test_accepted_close_with_positive_fill_is_incident_mismatch(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("8")),),
            orders=(
                order(
                    "close-accepted",
                    side="sell",
                    qty="13",
                    filled="5",
                    status="accepted",
                ),
            ),
            expected_close_context=expected_close(
                "close-accepted", remaining_quantity=Decimal("8")
            ),
        )

        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_CLOSE_FILL_STATUS_MISMATCH", result.issue_codes)

    def test_pending_new_close_with_positive_fill_is_incident_mismatch(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("8")),),
            orders=(
                order(
                    "close-pending-new",
                    side="sell",
                    qty="13",
                    filled="5",
                    status="pending_new",
                ),
            ),
            expected_close_context=expected_close(
                "close-pending-new", remaining_quantity=Decimal("8")
            ),
        )

        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_CLOSE_FILL_STATUS_MISMATCH", result.issue_codes)

    def test_new_close_with_positive_fill_is_incident_mismatch(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("8")),),
            orders=(
                order(
                    "close-new",
                    side="sell",
                    qty="13",
                    filled="5",
                    status="new",
                ),
            ),
            expected_close_context=expected_close(
                "close-new", remaining_quantity=Decimal("8")
            ),
        )

        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_CLOSE_FILL_STATUS_MISMATCH", result.issue_codes)

    def test_partially_filled_close_with_zero_fill_is_incident_mismatch(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(
                order(
                    "close-partial-zero",
                    side="sell",
                    qty="13",
                    filled="0",
                    status="partially_filled",
                ),
            ),
            expected_close_context=expected_close("close-partial-zero"),
        )

        self.assertEqual(result.status, ReconciliationStatus.INCIDENT_MISMATCH)
        self.assertIn("PENDING_CLOSE_FILL_STATUS_MISMATCH", result.issue_codes)

    def test_partially_filled_close_with_coherent_positive_fill_is_controlled(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("8")),),
            orders=(
                order(
                    "close-partial-valid",
                    side="sell",
                    qty="13",
                    filled="5",
                    status="partially_filled",
                ),
            ),
            expected_close_context=expected_close(
                "close-partial-valid", remaining_quantity=Decimal("8")
            ),
        )

        self.assertEqual(result.status, ReconciliationStatus.CONTROLLED_CLOSING)

    def test_new_close_with_zero_fill_and_exact_provenance_is_controlled(self):
        result = result_for(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(
                order(
                    "close-new-valid",
                    side="sell",
                    qty="13",
                    filled="0",
                    status="new",
                ),
            ),
            expected_close_context=expected_close("close-new-valid"),
        )

        self.assertEqual(result.status, ReconciliationStatus.CONTROLLED_CLOSING)

    def test_query_failures_incompleteness_and_caps_block_immediately(self):
        cases = (
            {"position_state": BrokerQueryState.UNKNOWN},
            {"order_state": BrokerQueryState.INCOMPLETE},
            {"positions_complete": False},
            {"orders_at_cap": True},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                result = result_for(**kwargs)
                self.assertEqual(result.status, ReconciliationStatus.BLOCKED_UNKNOWN_TRANSIENT)
                self.assertFalse(result.entry_allowed)
                self.assertFalse(result.incident_required)
                self.assertTrue(result.auto_recoverable)

    def test_malformed_completeness_or_order_metadata_cannot_be_treated_as_healthy(self):
        with self.assertRaises(BrokerNormalizationError):
            BrokerSnapshot(
                positions=(),
                orders=(),
                position_query_state=BrokerQueryState.OK,
                order_query_state=BrokerQueryState.OK,
                positions_complete="false",
                orders_complete=True,
                observed_at=T0,
            )
        with self.assertRaises(BrokerNormalizationError):
            normalize_order(
                {
                    "id": "malformed-stop",
                    "symbol": "SPY",
                    "side": "sell",
                    "qty": "13",
                    "filled_qty": "0",
                    "status": "new",
                    "type": "stop",
                }
            )

    def test_semantic_hash_ignores_observation_time_and_decimal_formatting(self):
        first = snapshot(
            positions=(normalize_position({"symbol": "spy", "qty": "13.000", "side": "long"}),),
            orders=(stop("stable", qty="13.0"),),
            observed_at="2026-07-14T14:00:00Z",
        )
        second = snapshot(
            positions=(PositionView("SPY", Decimal("13")),),
            orders=(stop("stable", qty=Decimal("13.0000")),),
            observed_at="2026-07-14T15:00:00+00:00",
        )
        self.assertEqual(first.snapshot_hash, second.snapshot_hash)

    def test_semantic_hash_ignores_order_refresh_timestamps(self):
        first = snapshot(
            orders=(stop("stable-time", stop_price="400",),),
            observed_at=T0,
        )
        second = snapshot(
            orders=(
                order(
                    "stable-time",
                    side="sell",
                    order_type="stop",
                    order_class="bracket",
                    parent_id="parent",
                    stop_price="400",
                    submitted_at=T0 + timedelta(hours=1),
                ),
            ),
            observed_at=T0 + timedelta(hours=2),
        )
        self.assertNotEqual(first.orders[0].submitted_at, second.orders[0].submitted_at)
        self.assertNotEqual(first.orders[0].updated_at, second.orders[0].updated_at)
        self.assertEqual(first.snapshot_hash, second.snapshot_hash)

    def test_semantic_hash_changes_with_order_status_quantity_and_prices(self):
        baseline_stop = snapshot(orders=(stop("semantic", stop_price="400"),))
        variants = {
            "status": snapshot(orders=(stop("semantic", status="held", stop_price="400"),)),
            "quantity": snapshot(orders=(stop("semantic", qty="12", stop_price="400"),)),
            "stop_price": snapshot(orders=(stop("semantic", stop_price="399"),)),
        }
        for field, variant in variants.items():
            with self.subTest(field=field):
                self.assertNotEqual(baseline_stop.snapshot_hash, variant.snapshot_hash)
        baseline_stop_limit = snapshot(
            orders=(
                stop(
                    "semantic-limit",
                    order_type="stop_limit",
                    stop_price="400",
                    limit_price="399",
                ),
            )
        )
        changed_limit = snapshot(
            orders=(
                stop(
                    "semantic-limit",
                    order_type="stop_limit",
                    stop_price="400",
                    limit_price="398",
                ),
            )
        )
        self.assertNotEqual(baseline_stop_limit.snapshot_hash, changed_limit.snapshot_hash)

    def test_two_equivalent_healthy_observations_recover_transient_block(self):
        state = apply_safety_state_transition(None, unknown(T0), observed_at=T0)
        healthy = healthy_flat(T0 + timedelta(seconds=1))
        state = apply_safety_state_transition(state, healthy, observed_at=T0 + timedelta(seconds=1))
        self.assertTrue(state.transient_blocked)
        state = apply_safety_state_transition(state, healthy, observed_at=T0 + timedelta(seconds=2))
        self.assertEqual(
            state.consecutive_healthy_recovery_count, REQUIRED_HEALTHY_RECOVERY_SNAPSHOTS
        )
        self.assertFalse(state.transient_blocked)
        self.assertIsNone(state.incident)
        self.assertEqual(state.total_unknown_observation_count, 1)
        self.assertEqual(state.last_unknown_at, T0)
        self.assertIsNotNone(state.last_unknown_snapshot_hash)
        self.assertIn("POSITIONS_QUERY_UNKNOWN", state.last_unknown_issue_codes)

    def test_entry_authorization_fails_closed_without_complete_evidence(self):
        healthy = reconcile_snapshot(
            snapshot(), holding_policy=HoldingPolicy.INTRADAY_FLAT
        )
        clean_state = SafetyState()
        current_state = apply_safety_state_transition(None, healthy, observed_at=T0)
        complete = {
            "holding_policy": HoldingPolicy.INTRADAY_FLAT,
            "safety_state": current_state,
            "pre_submit_snapshot_hash": healthy.snapshot_hash,
            "policy_authorized": True,
        }
        self.assertFalse(healthy.entry_allowed)
        self.assertFalse(
            can_submit_entry(
                healthy,
                holding_policy=HoldingPolicy.INTRADAY_FLAT,
                safety_state=clean_state,
                pre_submit_snapshot_hash=healthy.snapshot_hash,
                policy_authorized=True,
            )
        )
        self.assertFalse(
            can_submit_entry(
                healthy,
                holding_policy=HoldingPolicy.INTRADAY_FLAT,
                pre_submit_snapshot_hash=healthy.snapshot_hash,
                policy_authorized=True,
            )
        )
        self.assertFalse(
            can_submit_entry(
                healthy,
                holding_policy=HoldingPolicy.INTRADAY_FLAT,
                safety_state=current_state,
                pre_submit_snapshot_hash=healthy.snapshot_hash,
            )
        )
        self.assertFalse(
            can_submit_entry(
                healthy,
                holding_policy=HoldingPolicy.INTRADAY_FLAT,
                safety_state=current_state,
                policy_authorized=True,
            )
        )
        self.assertFalse(
            can_submit_entry(
                healthy,
                **{
                    **complete,
                    "pre_submit_snapshot_hash": "mismatched-snapshot",
                },
            )
        )
        self.assertTrue(can_submit_entry(healthy, **complete))

    def test_entry_authorization_rejects_stale_or_nonhealthy_safety_evidence(self):
        healthy = reconcile_snapshot(
            snapshot(), holding_policy=HoldingPolicy.INTRADAY_FLAT
        )
        current = apply_safety_state_transition(None, healthy, observed_at=T0)
        common = {
            "holding_policy": HoldingPolicy.INTRADAY_FLAT,
            "pre_submit_snapshot_hash": healthy.snapshot_hash,
            "policy_authorized": True,
        }
        stale = replace(current, last_snapshot_hash="stale-snapshot")
        nonhealthy = replace(
            current, last_status=ReconciliationStatus.BLOCKED_UNKNOWN_TRANSIENT
        )
        undated = replace(current, last_evidence_at=None)
        self.assertFalse(can_submit_entry(healthy, safety_state=stale, **common))
        self.assertFalse(can_submit_entry(healthy, safety_state=nonhealthy, **common))
        self.assertFalse(can_submit_entry(healthy, safety_state=undated, **common))

    def test_entry_authorization_rejects_open_and_clearable_incidents(self):
        healthy = healthy_flat(T0 + timedelta(seconds=1))
        unsafe = result_for(positions=(PositionView("SPY", Decimal("-13")),))
        open_state = apply_safety_state_transition(None, unsafe, observed_at=T0)
        clearable_state = apply_safety_state_transition(
            open_state, healthy, observed_at=T0 + timedelta(seconds=1)
        )
        clearable_state = apply_safety_state_transition(
            clearable_state, healthy, observed_at=T0 + timedelta(seconds=2)
        )
        self.assertEqual(open_state.incident.state, SafetyIncidentState.OPEN)
        self.assertEqual(clearable_state.incident.state, SafetyIncidentState.CLEARABLE)
        for state in (open_state, clearable_state):
            with self.subTest(incident_state=state.incident.state):
                self.assertFalse(
                    can_submit_entry(
                        healthy,
                        holding_policy=HoldingPolicy.INTRADAY_FLAT,
                        safety_state=state,
                        pre_submit_snapshot_hash=healthy.snapshot_hash,
                        policy_authorized=True,
                    )
                )

    def test_persistent_incident_becomes_clearable_but_never_auto_clears(self):
        unsafe = result_for(positions=(PositionView("SPY", Decimal("-13")),))
        state = apply_safety_state_transition(None, unsafe, observed_at=T0)
        self.assertEqual(state.incident.state, SafetyIncidentState.OPEN)
        healthy = healthy_flat(T0 + timedelta(seconds=1))
        for seconds in (1, 2, 3):
            state = apply_safety_state_transition(
                state, healthy, observed_at=T0 + timedelta(seconds=seconds)
            )
        self.assertEqual(state.incident.state, SafetyIncidentState.CLEARABLE)
        self.assertTrue(state.transient_blocked)

    def test_unknown_observation_reopens_a_clearable_incident(self):
        unsafe = result_for(positions=(PositionView("SPY", Decimal("-13")),))
        state = apply_safety_state_transition(None, unsafe, observed_at=T0)
        healthy = healthy_flat(T0 + timedelta(seconds=1))
        state = apply_safety_state_transition(state, healthy, observed_at=T0 + timedelta(seconds=1))
        state = apply_safety_state_transition(state, healthy, observed_at=T0 + timedelta(seconds=2))
        self.assertEqual(state.incident.state, SafetyIncidentState.CLEARABLE)
        state = apply_safety_state_transition(
            state, unknown(T0 + timedelta(seconds=3)), observed_at=T0 + timedelta(seconds=3)
        )
        self.assertEqual(state.incident.state, SafetyIncidentState.OPEN)

    def test_repeated_unknown_below_threshold_remains_transient(self):
        state = None
        for offset in range(MAX_CONSECUTIVE_UNKNOWN_SNAPSHOTS - 1):
            at = T0 + timedelta(seconds=offset)
            state = apply_safety_state_transition(state, unknown(at), observed_at=at)
        self.assertTrue(state.transient_blocked)
        self.assertIsNone(state.incident)

    def test_unknown_opens_incident_only_after_count_and_elapsed_thresholds(self):
        state = None
        for at in (
            T0,
            T0 + timedelta(seconds=60),
            T0 + timedelta(seconds=MAX_UNKNOWN_ELAPSED_SECONDS),
        ):
            state = apply_safety_state_transition(state, unknown(at), observed_at=at)
        self.assertEqual(state.consecutive_unknown_count, MAX_CONSECUTIVE_UNKNOWN_SNAPSHOTS)
        self.assertEqual(state.incident.state, SafetyIncidentState.OPEN)
        self.assertIn("BROKER_STATE_UNKNOWN_PROLONGED", state.incident.reason_codes)

    def test_unknown_count_without_elapsed_threshold_stays_transient(self):
        state = None
        for offset in range(MAX_CONSECUTIVE_UNKNOWN_SNAPSHOTS):
            at = T0 + timedelta(seconds=offset)
            state = apply_safety_state_transition(state, unknown(at), observed_at=at)
        self.assertIsNone(state.incident)

    def test_unknown_elapsed_without_count_threshold_stays_transient(self):
        state = apply_safety_state_transition(None, unknown(T0), observed_at=T0)
        later = T0 + timedelta(seconds=MAX_UNKNOWN_ELAPSED_SECONDS)
        state = apply_safety_state_transition(state, unknown(later), observed_at=later)
        self.assertEqual(state.consecutive_unknown_count, 2)
        self.assertIsNone(state.incident)

    def test_non_strategy_account_exposure_blocks_spy(self):
        cases = (
            ((PositionView("QQQ", Decimal("1")),), ()),
            ((), (order("qqq-order", symbol="QQQ"),)),
        )
        for positions, orders in cases:
            with self.subTest(positions=positions, orders=orders):
                result = result_for(positions=positions, orders=orders)
                self.assertEqual(result.status, ReconciliationStatus.INCIDENT_UNEXPECTED_EXPOSURE)
                self.assertEqual(result.severity, Severity.CRITICAL)
                self.assertFalse(result.entry_allowed)

    def test_fractional_decimal_quantities_are_not_truncated(self):
        position = normalize_position({"symbol": "SPY", "qty": "0.125", "side": "long"})
        result = result_for(positions=(position,), orders=(stop("fractional", qty="0.125"),))
        self.assertEqual(position.quantity, Decimal("0.125"))
        self.assertEqual(result.status, ReconciliationStatus.HEALTHY_PROTECTED)
        self.assertEqual(result.coverage_summaries[0].required_quantity, Decimal("0.125"))
        self.assertEqual(
            result.coverage_summaries[0].valid_active_stop_quantity, Decimal("0.125")
        )


if __name__ == "__main__":
    unittest.main()
