"""Pure broker-state normalization and PAPER_LIVE safety reconciliation.

This module deliberately has no Alpaca imports, credentials, network access, or
broker mutation capability.  It converts broker-shaped values into exact,
immutable views and classifies the complete account state supplied by a caller.

Holding policy is intentionally unresolved in this patch.  Reconciliation may
classify broker state as healthy, but entry permission remains false until a
separate, approved policy change selects a holding policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


REQUIRED_HEALTHY_RECOVERY_SNAPSHOTS = 2
MAX_CONSECUTIVE_UNKNOWN_SNAPSHOTS = 3
MAX_UNKNOWN_ELAPSED_SECONDS = 120
PENDING_ORDER_STALE_AFTER_SECONDS = 300


class BrokerQueryState(str, Enum):
    OK = "OK"
    UNKNOWN = "UNKNOWN"
    INCOMPLETE = "INCOMPLETE"


class HoldingPolicy(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    INTRADAY_FLAT = "INTRADAY_FLAT"
    OVERNIGHT_PROTECTED = "OVERNIGHT_PROTECTED"
    HYBRID = "HYBRID"


DEFAULT_HOLDING_POLICY = HoldingPolicy.UNRESOLVED


class ReconciliationStatus(str, Enum):
    HEALTHY_FLAT = "HEALTHY_FLAT"
    HEALTHY_PROTECTED = "HEALTHY_PROTECTED"
    CONTROLLED_ENTRY_PENDING = "CONTROLLED_ENTRY_PENDING"
    CONTROLLED_CANCELLING = "CONTROLLED_CANCELLING"
    CONTROLLED_FLATTENING = "CONTROLLED_FLATTENING"
    CONTROLLED_CLOSING = "CONTROLLED_CLOSING"
    BLOCKED_FLAT_WITH_UNRESOLVED_ORDERS = "BLOCKED_FLAT_WITH_UNRESOLVED_ORDERS"
    BLOCKED_POLICY_UNRESOLVED = "BLOCKED_POLICY_UNRESOLVED"
    BLOCKED_UNKNOWN_TRANSIENT = "BLOCKED_UNKNOWN_TRANSIENT"
    INCIDENT_UNPROTECTED = "INCIDENT_UNPROTECTED"
    INCIDENT_UNDERCOVERED = "INCIDENT_UNDERCOVERED"
    INCIDENT_OVERCOVERED = "INCIDENT_OVERCOVERED"
    INCIDENT_WRONG_SIDE = "INCIDENT_WRONG_SIDE"
    INCIDENT_MISMATCH = "INCIDENT_MISMATCH"
    INCIDENT_UNEXPECTED_EXPOSURE = "INCIDENT_UNEXPECTED_EXPOSURE"
    INCIDENT_STATE_CORRUPT = "INCIDENT_STATE_CORRUPT"
    UNKNOWN = "UNKNOWN"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SafetyIncidentState(str, Enum):
    OPEN = "OPEN"
    CLEARABLE = "CLEARABLE"
    CLEARED = "CLEARED"


class BrokerNormalizationError(ValueError):
    """A broker value cannot be represented safely and exactly."""


@dataclass(frozen=True)
class PositionView:
    symbol: str
    quantity: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _required_text(self.symbol, "position symbol").upper())
        object.__setattr__(self, "quantity", _decimal(self.quantity, "position quantity"))


@dataclass(frozen=True)
class OrderView:
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: str
    original_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    status: str
    order_type: str
    order_class: str
    time_in_force: str
    parent_id: Optional[str] = None
    replacement_ids: tuple[str, ...] = ()
    group_id: Optional[str] = None
    nested_level: int = 0
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    trail_price: Optional[Decimal] = None
    trail_percent: Optional[Decimal] = None
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _required_text(self.order_id, "order id"))
        object.__setattr__(self, "client_order_id", _optional_text(self.client_order_id))
        object.__setattr__(self, "symbol", _required_text(self.symbol, "order symbol").upper())
        object.__setattr__(self, "side", _enum_text(self.side).lower())
        if self.side not in {"buy", "sell"}:
            raise BrokerNormalizationError(f"unsupported order side: {self.side!r}")
        original = _decimal(self.original_quantity, "order quantity")
        filled = _decimal(self.filled_quantity, "filled quantity")
        remaining = _decimal(self.remaining_quantity, "remaining quantity")
        if original < 0 or filled < 0 or remaining < 0 or filled > original:
            raise BrokerNormalizationError("order quantities must satisfy 0 <= filled <= original")
        if remaining != original - filled:
            raise BrokerNormalizationError("remaining quantity does not equal original minus filled")
        object.__setattr__(self, "original_quantity", original)
        object.__setattr__(self, "filled_quantity", filled)
        object.__setattr__(self, "remaining_quantity", remaining)
        object.__setattr__(self, "status", _required_text(self.status, "order status").lower())
        object.__setattr__(self, "order_type", _required_text(self.order_type, "order type").lower())
        object.__setattr__(self, "order_class", _enum_text(self.order_class or "simple").lower())
        object.__setattr__(
            self,
            "time_in_force",
            _required_text(self.time_in_force, "order time in force").lower(),
        )
        object.__setattr__(self, "parent_id", _optional_text(self.parent_id))
        object.__setattr__(self, "group_id", _optional_text(self.group_id) or self.parent_id or self.order_id)
        object.__setattr__(self, "replacement_ids", tuple(sorted(filter(None, map(_optional_text, self.replacement_ids)))))
        object.__setattr__(self, "nested_level", int(self.nested_level))
        for name in ("limit_price", "stop_price", "trail_price", "trail_percent"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else _decimal(value, name))
        object.__setattr__(self, "submitted_at", _datetime(self.submitted_at))
        object.__setattr__(self, "updated_at", _datetime(self.updated_at))


@dataclass(frozen=True)
class BrokerSnapshot:
    positions: tuple[PositionView, ...]
    orders: tuple[OrderView, ...]
    position_query_state: BrokerQueryState
    order_query_state: BrokerQueryState
    positions_complete: bool
    orders_complete: bool
    observed_at: datetime
    positions_at_known_cap: bool = False
    orders_at_known_cap: bool = False
    snapshot_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", tuple(self.positions))
        object.__setattr__(self, "orders", tuple(self.orders))
        if any(not isinstance(position, PositionView) for position in self.positions):
            raise BrokerNormalizationError("snapshot positions must be normalized PositionView values")
        if any(not isinstance(order, OrderView) for order in self.orders):
            raise BrokerNormalizationError("snapshot orders must be normalized OrderView values")
        for name in (
            "positions_complete",
            "orders_complete",
            "positions_at_known_cap",
            "orders_at_known_cap",
        ):
            if type(getattr(self, name)) is not bool:
                raise BrokerNormalizationError(f"snapshot {name} must be boolean")
        object.__setattr__(self, "position_query_state", BrokerQueryState(self.position_query_state))
        object.__setattr__(self, "order_query_state", BrokerQueryState(self.order_query_state))
        object.__setattr__(self, "observed_at", _datetime(self.observed_at, required=True))
        computed = semantic_snapshot_hash(self)
        if self.snapshot_hash and self.snapshot_hash != computed:
            raise BrokerNormalizationError("supplied snapshot hash does not match snapshot contents")
        object.__setattr__(self, "snapshot_hash", computed)


@dataclass(frozen=True)
class ExpectedEntryContext:
    """Exact caller-owned provenance for one entry submission in flight."""

    client_order_id: str
    symbol: str
    side: str
    quantity: Decimal
    order_class: str
    broker_order_id: Optional[str] = None
    order_type: Optional[str] = None
    time_in_force: Optional[str] = None
    stop_price: Optional[Decimal] = None
    take_profit_limit_price: Optional[Decimal] = None
    request_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "client_order_id",
            _required_text(self.client_order_id, "expected client order id"),
        )
        object.__setattr__(
            self,
            "symbol",
            _required_text(self.symbol, "expected entry symbol").upper(),
        )
        side = _required_text(self.side, "expected entry side").lower()
        if side not in {"buy", "sell"}:
            raise BrokerNormalizationError(f"unsupported expected entry side: {side!r}")
        object.__setattr__(self, "side", side)
        if not isinstance(self.quantity, Decimal):
            raise BrokerNormalizationError("expected entry quantity must be an exact Decimal")
        if not self.quantity.is_finite() or self.quantity <= 0:
            raise BrokerNormalizationError("expected entry quantity must be finite and positive")
        object.__setattr__(
            self,
            "order_class",
            _required_text(self.order_class, "expected entry order class").lower(),
        )
        object.__setattr__(self, "broker_order_id", _optional_text(self.broker_order_id))
        object.__setattr__(
            self,
            "order_type",
            None
            if self.order_type is None
            else _required_text(self.order_type, "expected entry order type").lower(),
        )
        object.__setattr__(
            self,
            "time_in_force",
            None
            if self.time_in_force is None
            else _required_text(self.time_in_force, "expected entry time in force").lower(),
        )
        for name in ("stop_price", "take_profit_limit_price"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                    raise BrokerNormalizationError(f"expected entry {name} must be an exact positive Decimal")
        fingerprint = _optional_text(self.request_fingerprint)
        if fingerprint is not None:
            digest = fingerprint.removeprefix("sha256:")
            if (
                not fingerprint.startswith("sha256:")
                or len(digest) != 64
                or digest.lower() != digest
                or any(ch not in "0123456789abcdef" for ch in digest)
            ):
                raise BrokerNormalizationError(
                    "expected entry request fingerprint must be lowercase SHA-256"
                )
        object.__setattr__(self, "request_fingerprint", fingerprint)


@dataclass(frozen=True)
class ExpectedCloseContext:
    """Exact caller-owned provenance for one immediate market close in flight."""

    client_order_id: str
    symbol: str
    side: str
    remaining_quantity: Decimal
    order_type: str
    broker_order_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "client_order_id",
            _required_text(self.client_order_id, "expected close client order id"),
        )
        object.__setattr__(
            self,
            "symbol",
            _required_text(self.symbol, "expected close symbol").upper(),
        )
        side = _required_text(self.side, "expected close side").lower()
        if side not in {"buy", "sell"}:
            raise BrokerNormalizationError(f"unsupported expected close side: {side!r}")
        object.__setattr__(self, "side", side)
        if not isinstance(self.remaining_quantity, Decimal):
            raise BrokerNormalizationError(
                "expected close remaining quantity must be an exact Decimal"
            )
        if not self.remaining_quantity.is_finite() or self.remaining_quantity <= 0:
            raise BrokerNormalizationError(
                "expected close remaining quantity must be finite and positive"
            )
        object.__setattr__(
            self,
            "order_type",
            _required_text(self.order_type, "expected close order type").lower(),
        )
        object.__setattr__(self, "broker_order_id", _optional_text(self.broker_order_id))


@dataclass(frozen=True)
class ExpectedCancellationContext:
    """Exact durable IDs that the flatten controller may be cancelling."""

    target_order_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        values = tuple(sorted({_required_text(item, "cancellation target id") for item in self.target_order_ids}))
        if not values:
            raise BrokerNormalizationError("expected cancellation targets are required")
        object.__setattr__(self, "target_order_ids", values)


@dataclass(frozen=True)
class CoverageSummary:
    symbol: str
    position_quantity: Decimal
    required_protective_side: str
    required_quantity: Decimal
    valid_active_stop_quantity: Decimal
    uncovered_quantity: Decimal
    overcovered_quantity: Decimal
    pending_closing_quantity: Decimal
    valid_protection_ids: tuple[str, ...] = ()
    invalid_order_ids: tuple[str, ...] = ()
    issue_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReconciliationResult:
    status: ReconciliationStatus
    severity: Severity
    issue_codes: tuple[str, ...]
    entry_allowed: bool
    incident_required: bool
    auto_recoverable: bool
    snapshot_hash: str
    coverage_summaries: tuple[CoverageSummary, ...] = ()
    operator_action_required: bool = False
    controlled_transition_order_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class IncidentClearMetadata:
    incident_id: str
    operator: str
    reason: str
    timestamp: datetime
    verification_snapshot_hashes: tuple[str, ...]


@dataclass(frozen=True)
class SafetyIncident:
    incident_id: str
    state: SafetyIncidentState
    reason_codes: tuple[str, ...]
    first_observed: datetime
    last_observed: datetime
    occurrence_count: int
    first_snapshot_hash: str
    latest_snapshot_hash: str
    clear_metadata: Optional[IncidentClearMetadata] = None


@dataclass(frozen=True)
class SafetyState:
    transient_blocked: bool = False
    consecutive_unknown_count: int = 0
    unknown_since: Optional[datetime] = None
    recovery_snapshot_hash: Optional[str] = None
    consecutive_healthy_recovery_count: int = 0
    incident: Optional[SafetyIncident] = None
    last_status: Optional[ReconciliationStatus] = None
    last_snapshot_hash: Optional[str] = None
    evidence_count: int = 0
    total_unknown_observation_count: int = 0
    last_unknown_at: Optional[datetime] = None
    last_unknown_snapshot_hash: Optional[str] = None
    last_unknown_issue_codes: tuple[str, ...] = ()
    last_evidence_at: Optional[datetime] = None


@dataclass(frozen=True)
class TransitionClassification:
    matched: bool
    unsafe: bool
    issue_codes: tuple[str, ...] = ()
    order_ids: tuple[str, ...] = ()
    stale: bool = False


ACTIVE_STOP_STATUSES = frozenset({"new", "partially_filled"})
PENDING_TRANSITION_STATUSES = frozenset({"accepted", "pending_new", "new", "partially_filled", "held"})
PENDING_ENTRY_ROOT_STATUSES = frozenset({"accepted", "pending_new", "new", "held"})
PENDING_CLOSE_LIVE_STATUSES = frozenset(
    {"accepted", "pending_new", "new", "partially_filled"}
)
PENDING_CLOSE_STALE_AFTER_SECONDS = 120
MAX_ORDER_TIMESTAMP_FUTURE_SKEW_SECONDS = 5
# A flat account may only accept bracket descendants that the broker explicitly
# reports as preactivation-only.  Active or terminal children are not coherent
# with a wholly unfilled parent.
PENDING_BRACKET_PREACTIVATION_STATUSES = frozenset({"held", "inactive"})
TERMINAL_OR_UNCERTAIN_PROTECTION_STATUSES = frozenset(
    {
        "canceled",
        "expired",
        "rejected",
        "suspended",
        "pending_cancel",
        "pending_replace",
        "replaced",
        "inactive",
        "held",
        "accepted",
        "pending_new",
        "stopped",
        "calculated",
        "done_for_day",
        "filled",
    }
)
STOP_ORDER_TYPES = frozenset({"stop", "stop_limit", "trailing_stop"})
HEALTHY_STATUSES = frozenset(
    {ReconciliationStatus.HEALTHY_FLAT, ReconciliationStatus.HEALTHY_PROTECTED}
)


def _read(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _enum_text(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    if value is None:
        return ""
    return str(value).strip()


def _optional_text(value: Any) -> Optional[str]:
    text = _enum_text(value)
    return text or None


def _required_text(value: Any, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise BrokerNormalizationError(f"missing {field_name}")
    return text


def _decimal(value: Any, field_name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise BrokerNormalizationError(f"invalid {field_name}: {value!r}")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise BrokerNormalizationError(f"invalid {field_name}: {value!r}") from exc
    if not decimal_value.is_finite():
        raise BrokerNormalizationError(f"non-finite {field_name}: {value!r}")
    return decimal_value


def _optional_decimal(value: Any, field_name: str) -> Optional[Decimal]:
    return None if value is None or _enum_text(value) == "" else _decimal(value, field_name)


def _datetime(value: Any, *, required: bool = False) -> Optional[datetime]:
    if value is None or value == "":
        if required:
            raise BrokerNormalizationError("missing required timestamp")
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise BrokerNormalizationError(f"invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return _datetime(value, required=True).isoformat().replace("+00:00", "Z")


def _decimal_text(value: Optional[Decimal]) -> Optional[str]:
    if value is None:
        return None
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def normalize_position(position: Any) -> PositionView:
    """Normalize an Alpaca-shaped or mapping-shaped position without rounding."""

    symbol = _read(position, "symbol")
    quantity = _decimal(_read(position, "qty", "quantity"), "position quantity")
    side = _enum_text(_read(position, "side", default="")).lower()
    if side == "short" and quantity > 0:
        quantity = -quantity
    elif side == "long" and quantity < 0:
        raise BrokerNormalizationError("long position reported with negative quantity")
    elif side not in {"", "long", "short"}:
        raise BrokerNormalizationError(f"unsupported position side: {side!r}")
    return PositionView(symbol=symbol, quantity=quantity)


def calculate_remaining_quantity(original_quantity: Any, filled_quantity: Any) -> Decimal:
    original = _decimal(original_quantity, "order quantity")
    filled = _decimal(filled_quantity, "filled quantity")
    if original < 0 or filled < 0 or filled > original:
        raise BrokerNormalizationError("order quantities must satisfy 0 <= filled <= original")
    return original - filled


def normalize_order(
    order: Any,
    *,
    parent_id: Optional[str] = None,
    group_id: Optional[str] = None,
    nested_level: int = 0,
) -> OrderView:
    """Normalize an order while preserving relationships and reconciliation facts."""

    original = _decimal(_read(order, "qty", "quantity", "original_quantity"), "order quantity")
    filled = _decimal(_read(order, "filled_qty", "filled_quantity", default="0"), "filled quantity")
    remaining_raw = _read(order, "remaining_qty", "remaining_quantity", default=None)
    remaining = (
        calculate_remaining_quantity(original, filled)
        if remaining_raw is None
        else _decimal(remaining_raw, "remaining quantity")
    )
    order_id = _required_text(_read(order, "id", "order_id"), "order id")
    explicit_parent = _optional_text(_read(order, "parent_order_id", "parent_id", default=None))
    replacements = {
        _optional_text(_read(order, "replaced_by", default=None)),
        _optional_text(_read(order, "replaces", default=None)),
    }
    replacements.update(
        _optional_text(item)
        for item in (_read(order, "replacement_ids", default=()) or ())
    )
    return OrderView(
        order_id=order_id,
        client_order_id=_read(order, "client_order_id", default=None),
        symbol=_read(order, "symbol"),
        side=_read(order, "side"),
        original_quantity=original,
        filled_quantity=filled,
        remaining_quantity=remaining,
        status=_read(order, "status", default=""),
        order_type=_read(order, "type", "order_type", default=""),
        order_class=_read(order, "order_class", default="simple") or "simple",
        time_in_force=_read(order, "time_in_force", default=""),
        parent_id=parent_id or explicit_parent,
        replacement_ids=tuple(item for item in replacements if item),
        group_id=group_id or _read(order, "group_id", default=None) or parent_id or explicit_parent or order_id,
        nested_level=nested_level,
        limit_price=_optional_decimal(_read(order, "limit_price", default=None), "limit price"),
        stop_price=_optional_decimal(_read(order, "stop_price", default=None), "stop price"),
        trail_price=_optional_decimal(_read(order, "trail_price", default=None), "trail price"),
        trail_percent=_optional_decimal(_read(order, "trail_percent", default=None), "trail percent"),
        submitted_at=_datetime(_read(order, "submitted_at", "created_at", default=None)),
        updated_at=_datetime(_read(order, "updated_at", default=None)),
    )


def flatten_order_tree(orders: Iterable[Any]) -> tuple[OrderView, ...]:
    """Flatten nested broker orders, preserving their parent and root group IDs."""

    flattened: list[OrderView] = []

    def visit(raw: Any, parent_id: Optional[str], group_id: Optional[str], level: int) -> None:
        view = normalize_order(raw, parent_id=parent_id, group_id=group_id, nested_level=level)
        flattened.append(view)
        legs = _read(raw, "legs", default=None) or ()
        for leg in legs:
            visit(leg, view.order_id, view.group_id or view.order_id, level + 1)

    for root in orders:
        visit(root, None, None, 0)
    return tuple(flattened)


def _position_payload(position: PositionView) -> dict[str, Any]:
    return {"symbol": position.symbol, "quantity": _decimal_text(position.quantity)}


def _order_payload(order: OrderView) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "client_order_id": order.client_order_id,
        "symbol": order.symbol,
        "side": order.side,
        "original_quantity": _decimal_text(order.original_quantity),
        "filled_quantity": _decimal_text(order.filled_quantity),
        "remaining_quantity": _decimal_text(order.remaining_quantity),
        "status": order.status,
        "order_type": order.order_type,
        "order_class": order.order_class,
        "time_in_force": order.time_in_force,
        "parent_id": order.parent_id,
        "replacement_ids": list(order.replacement_ids),
        "group_id": order.group_id,
        "nested_level": order.nested_level,
        "limit_price": _decimal_text(order.limit_price),
        "stop_price": _decimal_text(order.stop_price),
        "trail_price": _decimal_text(order.trail_price),
        "trail_percent": _decimal_text(order.trail_percent),
    }


def semantic_snapshot_hash(snapshot: BrokerSnapshot) -> str:
    """Return a stable hash of broker facts, excluding observation-time noise."""

    payload = {
        "positions": sorted(
            (_position_payload(position) for position in snapshot.positions),
            key=lambda item: (item["symbol"], item["quantity"]),
        ),
        "orders": sorted(
            (_order_payload(order) for order in snapshot.orders),
            key=lambda item: (item["symbol"], item["group_id"] or "", item["order_id"]),
        ),
        "position_query_state": BrokerQueryState(snapshot.position_query_state).value,
        "order_query_state": BrokerQueryState(snapshot.order_query_state).value,
        "positions_complete": bool(snapshot.positions_complete),
        "orders_complete": bool(snapshot.orders_complete),
        "positions_at_known_cap": bool(snapshot.positions_at_known_cap),
        "orders_at_known_cap": bool(snapshot.orders_at_known_cap),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_stop(order: OrderView) -> bool:
    return order.order_type in STOP_ORDER_TYPES


def _is_finite_positive_price(value: Optional[Decimal]) -> bool:
    return isinstance(value, Decimal) and value.is_finite() and value > 0


def _has_valid_stop_price_shape(order: OrderView) -> bool:
    """Return whether a normalized stop order has all type-specific prices."""

    if order.order_type == "stop":
        return _is_finite_positive_price(order.stop_price)
    if order.order_type == "stop_limit":
        return _is_finite_positive_price(order.stop_price) and _is_finite_positive_price(
            order.limit_price
        )
    if order.order_type == "trailing_stop":
        return any(
            _is_finite_positive_price(value)
            for value in (order.trail_price, order.trail_percent, order.stop_price)
        )
    return False


def _has_valid_take_profit_price_shape(order: OrderView) -> bool:
    return order.order_type == "limit" and _is_finite_positive_price(order.limit_price)


def _is_active_stop(order: OrderView) -> bool:
    return _is_stop(order) and order.status in ACTIVE_STOP_STATUSES and order.remaining_quantity > 0


def _is_simple_close_candidate(order: OrderView) -> bool:
    return (
        order.parent_id is None
        and order.order_class in {"", "simple"}
        and not _is_stop(order)
        and order.status in PENDING_TRANSITION_STATUSES
        and order.remaining_quantity > 0
    )


def _order_age_seconds(order: OrderView, observed_at: Optional[datetime]) -> Optional[float]:
    if order.submitted_at is None or observed_at is None:
        return None
    return (_datetime(observed_at, required=True) - order.submitted_at).total_seconds()


def calculate_protective_coverage(
    position: PositionView,
    orders: Sequence[OrderView],
) -> CoverageSummary:
    """Calculate exact active stop coverage for one broker position.

    Coverage is grouped by order group and only stop legs are counted.  A take
    profit in the same OCO/bracket group is therefore never additive coverage.
    """

    required_quantity = abs(position.quantity)
    required_side = "sell" if position.quantity > 0 else "buy"
    symbol_orders = tuple(order for order in orders if order.symbol == position.symbol)
    active_stops = tuple(order for order in symbol_orders if _is_active_stop(order))
    invalid_price_stops = tuple(
        order for order in active_stops if not _has_valid_stop_price_shape(order)
    )
    price_valid_stops = tuple(
        order for order in active_stops if _has_valid_stop_price_shape(order)
    )
    valid_stops = tuple(
        order for order in price_valid_stops if order.side == required_side
    )
    wrong_side_stops = tuple(
        order for order in price_valid_stops if order.side != required_side
    )

    stops_by_group: dict[str, list[OrderView]] = {}
    for order in valid_stops:
        stops_by_group.setdefault(order.group_id or order.order_id, []).append(order)

    coverage = Decimal("0")
    issue_codes: list[str] = []
    for group_orders in stops_by_group.values():
        coverage += max(order.remaining_quantity for order in group_orders)
        if len(group_orders) > 1:
            issue_codes.append("MULTIPLE_ACTIVE_STOPS_IN_GROUP")

    if wrong_side_stops:
        issue_codes.append("WRONG_SIDE_PROTECTION")
    if invalid_price_stops:
        issue_codes.append("PROTECTION_STOP_PRICE_INVALID")
    uncovered = max(required_quantity - coverage, Decimal("0"))
    overcovered = max(coverage - required_quantity, Decimal("0"))
    if coverage == 0:
        issue_codes.append("NO_ACTIVE_STOP_COVERAGE")
    elif uncovered > 0:
        issue_codes.append("PROTECTION_UNDERCOVERED")
    elif overcovered > 0:
        issue_codes.append("PROTECTION_OVERCOVERED")

    pending_close = sum(
        (
            order.remaining_quantity
            for order in symbol_orders
            if _is_simple_close_candidate(order) and order.side == required_side
        ),
        Decimal("0"),
    )
    valid_ids = tuple(sorted(order.order_id for order in valid_stops))
    invalid_ids = tuple(
        sorted(
            order.order_id
            for order in symbol_orders
            if order.order_id not in valid_ids
            and (
                _is_stop(order)
                or order.parent_id is not None
                or order.status in TERMINAL_OR_UNCERTAIN_PROTECTION_STATUSES
            )
        )
    )
    return CoverageSummary(
        symbol=position.symbol,
        position_quantity=position.quantity,
        required_protective_side=required_side.upper(),
        required_quantity=required_quantity,
        valid_active_stop_quantity=coverage,
        uncovered_quantity=uncovered,
        overcovered_quantity=overcovered,
        pending_closing_quantity=pending_close,
        valid_protection_ids=valid_ids,
        invalid_order_ids=invalid_ids,
        issue_codes=tuple(sorted(set(issue_codes))),
    )


def classify_pending_entry(
    orders: Sequence[OrderView],
    *,
    symbol: str = "SPY",
    observed_at: Optional[datetime] = None,
    expected_entry: Optional[ExpectedEntryContext] = None,
    broker_snapshot: Optional[BrokerSnapshot] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> TransitionClassification:
    """Match one flat-account order group to explicit submission provenance.

    A match requires the complete flat snapshot as well as expected-entry
    provenance, so direct callers cannot authorize based on order shape alone.
    """

    all_orders = tuple(orders)
    if not all_orders:
        return TransitionClassification(matched=False, unsafe=False)
    all_ids = tuple(sorted(order.order_id for order in all_orders))
    if expected_entry is None:
        return TransitionClassification(
            matched=False,
            unsafe=True,
            issue_codes=("EXPECTED_ENTRY_CONTEXT_MISSING",),
            order_ids=all_ids,
        )
    if not isinstance(expected_entry, ExpectedEntryContext):
        raise BrokerNormalizationError(
            "expected_entry must be an ExpectedEntryContext with exact Decimal quantity"
        )
    snapshot_proves_complete_flat = (
        isinstance(broker_snapshot, BrokerSnapshot)
        and broker_snapshot.orders == all_orders
        and broker_snapshot.position_query_state == BrokerQueryState.OK
        and broker_snapshot.order_query_state == BrokerQueryState.OK
        and broker_snapshot.positions_complete
        and broker_snapshot.orders_complete
        and not broker_snapshot.positions_at_known_cap
        and not broker_snapshot.orders_at_known_cap
        and all(position.quantity == 0 for position in broker_snapshot.positions)
    )
    if not snapshot_proves_complete_flat:
        return TransitionClassification(
            matched=False,
            unsafe=True,
            issue_codes=("PENDING_ENTRY_SNAPSHOT_UNVERIFIED",),
            order_ids=all_ids,
        )

    groups: dict[str, list[OrderView]] = {}
    for order in all_orders:
        groups.setdefault(order.group_id or order.parent_id or order.order_id, []).append(order)
    if len(groups) != 1:
        return TransitionClassification(
            matched=False,
            unsafe=True,
            issue_codes=("DUPLICATE_PENDING_ENTRY_GROUPS",),
            order_ids=all_ids,
        )

    group = next(iter(groups.values()))
    roots = [order for order in group if order.parent_id is None]
    if len(roots) != 1:
        return TransitionClassification(
            matched=False,
            unsafe=True,
            issue_codes=("INCOHERENT_PENDING_ENTRY_ROOT",),
            order_ids=all_ids,
        )

    root = roots[0]
    issue_codes: list[str] = []
    if root.client_order_id != expected_entry.client_order_id:
        issue_codes.append("EXPECTED_ENTRY_CLIENT_ID_MISMATCH")
    if (
        expected_entry.broker_order_id is not None
        and root.order_id != expected_entry.broker_order_id
    ):
        issue_codes.append("EXPECTED_ENTRY_ID_MISMATCH")
    if root.symbol != expected_entry.symbol or expected_entry.symbol != symbol.upper():
        issue_codes.append("EXPECTED_ENTRY_SYMBOL_MISMATCH")
    if root.side != expected_entry.side:
        issue_codes.append("EXPECTED_ENTRY_SIDE_MISMATCH")
    if root.original_quantity != expected_entry.quantity:
        issue_codes.append("EXPECTED_ENTRY_QUANTITY_MISMATCH")
    if root.order_class != expected_entry.order_class:
        issue_codes.append("EXPECTED_ENTRY_ORDER_CLASS_MISMATCH")
    if expected_entry.order_type is not None and root.order_type != expected_entry.order_type:
        issue_codes.append("EXPECTED_ENTRY_ORDER_TYPE_MISMATCH")
    if expected_entry.time_in_force is not None and root.time_in_force != expected_entry.time_in_force:
        issue_codes.append("EXPECTED_ENTRY_TIF_MISMATCH")

    if root.status == "partially_filled" or root.filled_quantity > 0:
        issue_codes.append("FLAT_WITH_PARTIALLY_FILLED_ENTRY")
    if root.filled_quantity != 0 or root.remaining_quantity != root.original_quantity:
        issue_codes.append("INCOHERENT_PENDING_ENTRY_QUANTITY_STATE")
    if root.status not in PENDING_ENTRY_ROOT_STATUSES:
        issue_codes.append("INCOHERENT_PENDING_ENTRY_STATUS")
    if root.order_type in STOP_ORDER_TYPES or root.side not in {"buy", "sell"}:
        issue_codes.append("INCOHERENT_PENDING_ENTRY_TYPE")

    children = tuple(order for order in group if order is not root)
    if root.order_class == "bracket":
        if children:
            stop_children = tuple(child for child in children if child.order_type in STOP_ORDER_TYPES)
            take_profit_children = tuple(child for child in children if child.order_type == "limit")
            if len(children) != 2 or len(stop_children) != 1 or len(take_profit_children) != 1:
                issue_codes.append("PENDING_BRACKET_STRUCTURE_INVALID")
            if len(stop_children) != 1:
                issue_codes.append("PENDING_BRACKET_STOP_INVALID")
            if len(take_profit_children) != 1:
                issue_codes.append("PENDING_BRACKET_TAKE_PROFIT_INVALID")
            if (
                len(stop_children) == 1
                and expected_entry.stop_price is not None
                and stop_children[0].stop_price != expected_entry.stop_price
            ):
                issue_codes.append("EXPECTED_ENTRY_STOP_PRICE_MISMATCH")
            if (
                len(take_profit_children) == 1
                and expected_entry.take_profit_limit_price is not None
                and take_profit_children[0].limit_price != expected_entry.take_profit_limit_price
            ):
                issue_codes.append("EXPECTED_ENTRY_TAKE_PROFIT_PRICE_MISMATCH")
            for child in children:
                component_code = (
                    "PENDING_BRACKET_STOP_INVALID"
                    if child.order_type in STOP_ORDER_TYPES
                    else "PENDING_BRACKET_TAKE_PROFIT_INVALID"
                    if child.order_type == "limit"
                    else "PENDING_BRACKET_STRUCTURE_INVALID"
                )
                coherent_child = (
                    child.parent_id == root.order_id
                    and child.symbol == root.symbol
                    and child.side != root.side
                    and child.order_class == root.order_class
                    and child.time_in_force == root.time_in_force
                    and child.original_quantity == root.original_quantity
                    and child.filled_quantity == 0
                    and child.remaining_quantity == child.original_quantity
                    and child.status in PENDING_BRACKET_PREACTIVATION_STATUSES
                )
                if child.order_type in STOP_ORDER_TYPES:
                    coherent_child = coherent_child and _has_valid_stop_price_shape(child)
                elif child.order_type == "limit":
                    coherent_child = coherent_child and _has_valid_take_profit_price_shape(child)
                if not coherent_child:
                    issue_codes.extend(("PENDING_BRACKET_STRUCTURE_INVALID", component_code))
    elif children:
        issue_codes.append("PENDING_BRACKET_STRUCTURE_INVALID")

    stale = any(
        age is not None and age >= PENDING_ORDER_STALE_AFTER_SECONDS
        for age in (_order_age_seconds(order, observed_at) for order in group)
    )
    if stale:
        issue_codes.append("PENDING_ENTRY_STALE_EVIDENCE")
    escalation_context = context or {}
    context_requires_escalation = bool(
        escalation_context.get("submission_ambiguous")
        or escalation_context.get("status_inconsistent")
        or escalation_context.get("duplicate_observation")
    )
    unsafe = any(code != "PENDING_ENTRY_STALE_EVIDENCE" for code in issue_codes) or context_requires_escalation
    if context_requires_escalation:
        issue_codes.append("PENDING_ENTRY_CONTEXT_ESCALATION")
    return TransitionClassification(
        matched=not unsafe,
        unsafe=unsafe,
        issue_codes=tuple(sorted(set(issue_codes))),
        order_ids=tuple(sorted(order.order_id for order in group)),
        stale=stale,
    )


def classify_pending_close(
    position: PositionView,
    orders: Sequence[OrderView],
    *,
    expected_close: Optional[ExpectedCloseContext] = None,
    broker_snapshot: Optional[BrokerSnapshot] = None,
) -> TransitionClassification:
    """Recognize one fresh, provenance-bound immediate market close."""

    all_orders = tuple(orders)
    close_roots = tuple(
        order
        for order in all_orders
        if order.symbol == position.symbol
        and order.parent_id is None
        and order.order_class in {"", "simple"}
        and order.remaining_quantity > 0
    )
    if not close_roots:
        return TransitionClassification(matched=False, unsafe=False)
    ids = tuple(sorted(order.order_id for order in close_roots))
    if len(close_roots) != 1:
        return TransitionClassification(
            matched=False,
            unsafe=True,
            issue_codes=("MULTIPLE_PENDING_CLOSE_ORDERS",),
            order_ids=ids,
        )

    order = close_roots[0]
    issues: list[str] = []
    authoritative_positions = (
        tuple(
            candidate
            for candidate in broker_snapshot.positions
            if candidate.quantity != 0
        )
        if isinstance(broker_snapshot, BrokerSnapshot)
        else ()
    )
    snapshot_is_authoritative = (
        isinstance(broker_snapshot, BrokerSnapshot)
        and broker_snapshot.orders == all_orders
        and broker_snapshot.position_query_state == BrokerQueryState.OK
        and broker_snapshot.order_query_state == BrokerQueryState.OK
        and broker_snapshot.positions_complete
        and broker_snapshot.orders_complete
        and not broker_snapshot.positions_at_known_cap
        and not broker_snapshot.orders_at_known_cap
        and authoritative_positions == (position,)
    )
    if not snapshot_is_authoritative:
        issues.append("PENDING_CLOSE_SNAPSHOT_UNVERIFIED")

    if expected_close is None:
        issues.append("EXPECTED_CLOSE_CONTEXT_MISSING")
    elif not isinstance(expected_close, ExpectedCloseContext):
        raise BrokerNormalizationError(
            "expected_close must be an ExpectedCloseContext with exact Decimal quantity"
        )
    else:
        if order.client_order_id != expected_close.client_order_id:
            issues.append("EXPECTED_CLOSE_CLIENT_ID_MISMATCH")
        if (
            expected_close.broker_order_id is not None
            and order.order_id != expected_close.broker_order_id
        ):
            issues.append("EXPECTED_CLOSE_ID_MISMATCH")
        if order.symbol != expected_close.symbol or position.symbol != expected_close.symbol:
            issues.append("EXPECTED_CLOSE_SYMBOL_MISMATCH")

    required_side = "sell" if position.quantity > 0 else "buy"
    if order.side != required_side or (
        expected_close is not None and expected_close.side != required_side
    ):
        issues.append("EXPECTED_CLOSE_SIDE_MISMATCH")
    required_quantity = abs(position.quantity)
    if (
        order.remaining_quantity != required_quantity
        or order.original_quantity - order.filled_quantity != order.remaining_quantity
        or (
            expected_close is not None
            and expected_close.remaining_quantity != required_quantity
        )
        or (
            expected_close is not None
            and expected_close.remaining_quantity != order.remaining_quantity
        )
    ):
        issues.append("EXPECTED_CLOSE_QUANTITY_MISMATCH")
    if (
        order.order_type != "market"
        or (
            expected_close is not None
            and (
                expected_close.order_type != "market"
                or expected_close.order_type != order.order_type
            )
        )
    ):
        issues.append("EXPECTED_CLOSE_TYPE_MISMATCH")
    if order.status not in PENDING_CLOSE_LIVE_STATUSES:
        issues.append("PENDING_CLOSE_STATUS_INVALID")
    if (
        order.status in {"accepted", "pending_new", "new"}
        and order.filled_quantity != 0
    ) or (
        order.status == "partially_filled"
        and not (0 < order.filled_quantity < order.original_quantity)
    ):
        issues.append("PENDING_CLOSE_FILL_STATUS_MISMATCH")
    if order.submitted_at is None:
        issues.append("PENDING_CLOSE_TIMESTAMP_MISSING")
    elif snapshot_is_authoritative:
        age = _order_age_seconds(order, broker_snapshot.observed_at)
        if age is not None:
            if age < -MAX_ORDER_TIMESTAMP_FUTURE_SKEW_SECONDS:
                issues.append("PENDING_CLOSE_TIMESTAMP_FUTURE")
            elif age > PENDING_CLOSE_STALE_AFTER_SECONDS:
                issues.append("PENDING_CLOSE_STALE")

    other_active = tuple(
        candidate
        for candidate in all_orders
        if candidate.symbol == position.symbol
        and candidate.order_id != order.order_id
        and candidate.remaining_quantity > 0
        and candidate.status
        not in {"canceled", "expired", "rejected", "filled", "replaced", "done_for_day", "stopped"}
    )
    if other_active:
        issues.append("PENDING_CLOSE_WITH_OTHER_ACTIVE_ORDERS")
    if issues:
        return TransitionClassification(
            matched=False,
            unsafe=True,
            issue_codes=tuple(issues),
            order_ids=ids,
        )
    return TransitionClassification(matched=True, unsafe=False, order_ids=ids)


def _result(
    snapshot: BrokerSnapshot,
    status: ReconciliationStatus,
    severity: Severity,
    issue_codes: Iterable[str],
    *,
    holding_policy: HoldingPolicy,
    incident_required: bool,
    auto_recoverable: bool,
    coverage: Sequence[CoverageSummary] = (),
    operator_action_required: bool = False,
    controlled_ids: Sequence[str] = (),
) -> ReconciliationResult:
    base = ReconciliationResult(
        status=status,
        severity=severity,
        issue_codes=tuple(sorted(set(issue_codes))),
        entry_allowed=False,
        incident_required=incident_required,
        auto_recoverable=auto_recoverable,
        snapshot_hash=snapshot.snapshot_hash,
        coverage_summaries=tuple(coverage),
        operator_action_required=operator_action_required,
        controlled_transition_order_ids=tuple(sorted(set(controlled_ids))),
    )
    # Reconciliation classifies broker state only.  Final authorization requires
    # independently supplied persisted safety and policy evidence.
    return base


def reconcile_snapshot(
    snapshot: BrokerSnapshot,
    *,
    strategy_symbol: str = "SPY",
    holding_policy: HoldingPolicy = DEFAULT_HOLDING_POLICY,
    expected_entry: Optional[ExpectedEntryContext] = None,
    expected_close: Optional[ExpectedCloseContext] = None,
    expected_cancellations: Optional[ExpectedCancellationContext] = None,
    transition_context: Optional[Mapping[str, Any]] = None,
) -> ReconciliationResult:
    """Classify an authoritative, account-wide broker snapshot."""

    holding_policy = HoldingPolicy(holding_policy)
    transition_context = transition_context or {}
    if transition_context.get("submission_operator_action_required"):
        return _result(
            snapshot,
            ReconciliationStatus.INCIDENT_MISMATCH,
            Severity.CRITICAL,
            ("SUBMISSION_OPERATOR_ACTION_REQUIRED",),
            holding_policy=holding_policy,
            incident_required=True,
            auto_recoverable=False,
            operator_action_required=True,
        )
    if transition_context.get("submission_ambiguous_unresolved"):
        return _result(
            snapshot,
            ReconciliationStatus.BLOCKED_UNKNOWN_TRANSIENT,
            Severity.HIGH,
            ("SUBMISSION_AMBIGUITY_UNRESOLVED",),
            holding_policy=holding_policy,
            incident_required=False,
            auto_recoverable=True,
            operator_action_required=False,
        )
    if transition_context.get("flattening_operator_action_required"):
        return _result(
            snapshot,
            ReconciliationStatus.INCIDENT_MISMATCH,
            Severity.CRITICAL,
            ("FLATTENING_OPERATOR_ACTION_REQUIRED",),
            holding_policy=holding_policy,
            incident_required=True,
            auto_recoverable=False,
            operator_action_required=True,
        )
    if transition_context.get("flattening_ambiguous_unresolved"):
        return _result(
            snapshot,
            ReconciliationStatus.BLOCKED_UNKNOWN_TRANSIENT,
            Severity.HIGH,
            ("FLATTENING_AMBIGUITY_UNRESOLVED",),
            holding_policy=holding_policy,
            incident_required=False,
            auto_recoverable=True,
        )
    query_issues: list[str] = []
    if snapshot.position_query_state != BrokerQueryState.OK:
        query_issues.append(f"POSITIONS_QUERY_{snapshot.position_query_state.value}")
    if snapshot.order_query_state != BrokerQueryState.OK:
        query_issues.append(f"ORDERS_QUERY_{snapshot.order_query_state.value}")
    if not snapshot.positions_complete:
        query_issues.append("POSITIONS_RESPONSE_INCOMPLETE")
    if not snapshot.orders_complete:
        query_issues.append("ORDERS_RESPONSE_INCOMPLETE")
    if snapshot.positions_at_known_cap:
        query_issues.append("POSITIONS_RESPONSE_AT_KNOWN_CAP")
    if snapshot.orders_at_known_cap:
        query_issues.append("ORDERS_RESPONSE_AT_KNOWN_CAP")
    if query_issues:
        return _result(
            snapshot,
            ReconciliationStatus.BLOCKED_UNKNOWN_TRANSIENT,
            Severity.HIGH,
            query_issues,
            holding_policy=holding_policy,
            incident_required=False,
            auto_recoverable=True,
        )

    symbol = strategy_symbol.upper()
    positions = tuple(position for position in snapshot.positions if position.quantity != 0)
    unexpected_positions = tuple(position for position in positions if position.symbol != symbol)
    unexpected_orders = tuple(order for order in snapshot.orders if order.symbol != symbol)
    if unexpected_positions or unexpected_orders:
        codes = ["UNEXPECTED_ACCOUNT_POSITION"] if unexpected_positions else []
        if unexpected_orders:
            codes.append("UNEXPECTED_ACCOUNT_ORDER")
        return _result(
            snapshot,
            ReconciliationStatus.INCIDENT_UNEXPECTED_EXPOSURE,
            Severity.CRITICAL,
            codes,
            holding_policy=holding_policy,
            incident_required=True,
            auto_recoverable=False,
            operator_action_required=True,
        )

    if expected_cancellations is not None and not isinstance(expected_cancellations, ExpectedCancellationContext):
        raise BrokerNormalizationError("expected_cancellations must be ExpectedCancellationContext")
    active_order_ids = tuple(
        sorted(
            order.order_id for order in snapshot.orders
            if order.remaining_quantity > 0
            and order.status not in {"canceled", "expired", "rejected", "filled", "replaced", "done_for_day", "stopped"}
        )
    )
    if expected_cancellations is not None and active_order_ids:
        expected_ids = set(expected_cancellations.target_order_ids)
        if set(active_order_ids).issubset(expected_ids):
            return _result(
                snapshot,
                ReconciliationStatus.CONTROLLED_CANCELLING,
                Severity.WARNING,
                ("PENDING_CANCELLATION_MATCHES_PROVENANCE",),
                holding_policy=holding_policy,
                incident_required=False,
                auto_recoverable=True,
                controlled_ids=active_order_ids,
            )

    strategy_positions = tuple(position for position in positions if position.symbol == symbol)
    if len(strategy_positions) > 1:
        return _result(
            snapshot,
            ReconciliationStatus.INCIDENT_MISMATCH,
            Severity.CRITICAL,
            ("MULTIPLE_POSITION_RECORDS",),
            holding_policy=holding_policy,
            incident_required=True,
            auto_recoverable=False,
            operator_action_required=True,
        )

    if not strategy_positions:
        if not snapshot.orders:
            return _result(
                snapshot,
                ReconciliationStatus.HEALTHY_FLAT,
                Severity.INFO,
                (),
                holding_policy=holding_policy,
                incident_required=False,
                auto_recoverable=True,
            )
        pending = classify_pending_entry(
            snapshot.orders,
            symbol=symbol,
            observed_at=snapshot.observed_at,
            expected_entry=expected_entry,
            broker_snapshot=snapshot,
            context=transition_context,
        )
        if pending.matched:
            return _result(
                snapshot,
                ReconciliationStatus.CONTROLLED_ENTRY_PENDING,
                Severity.WARNING,
                pending.issue_codes,
                holding_policy=holding_policy,
                incident_required=False,
                auto_recoverable=True,
                controlled_ids=pending.order_ids,
            )
        if pending.unsafe:
            return _result(
                snapshot,
                ReconciliationStatus.INCIDENT_MISMATCH,
                Severity.CRITICAL,
                pending.issue_codes,
                holding_policy=holding_policy,
                incident_required=True,
                auto_recoverable=False,
                operator_action_required=True,
                controlled_ids=pending.order_ids,
            )
        return _result(
            snapshot,
            ReconciliationStatus.BLOCKED_FLAT_WITH_UNRESOLVED_ORDERS,
            Severity.HIGH,
            ("FLAT_WITH_UNRESOLVED_ORDERS",),
            holding_policy=holding_policy,
            incident_required=True,
            auto_recoverable=False,
            operator_action_required=True,
        )

    position = strategy_positions[0]
    pending_close = classify_pending_close(
        position,
        snapshot.orders,
        expected_close=expected_close,
        broker_snapshot=snapshot,
    )
    if pending_close.unsafe:
        return _result(
            snapshot,
            ReconciliationStatus.INCIDENT_MISMATCH,
            Severity.CRITICAL,
            pending_close.issue_codes,
            holding_policy=holding_policy,
            incident_required=True,
            auto_recoverable=False,
            operator_action_required=True,
            controlled_ids=pending_close.order_ids,
        )
    if pending_close.matched:
        return _result(
            snapshot,
            ReconciliationStatus.CONTROLLED_CLOSING,
            Severity.WARNING,
            ("PENDING_CLOSE_MATCHES_EXPOSURE",),
            holding_policy=holding_policy,
            incident_required=False,
            auto_recoverable=True,
            controlled_ids=pending_close.order_ids,
        )

    if transition_context.get("flattening_unresolved_close_recovery"):
        if active_order_ids:
            return _result(
                snapshot,
                ReconciliationStatus.INCIDENT_MISMATCH,
                Severity.CRITICAL,
                ("UNRESOLVED_CLOSE_WITH_UNEXPECTED_ACTIVE_ORDER",),
                holding_policy=holding_policy,
                incident_required=True,
                auto_recoverable=False,
                operator_action_required=True,
                controlled_ids=active_order_ids,
            )
        return _result(
            snapshot,
            ReconciliationStatus.CONTROLLED_FLATTENING,
            Severity.WARNING,
            ("UNRESOLVED_CLOSE_RECOVERY",),
            holding_policy=holding_policy,
            incident_required=False,
            auto_recoverable=True,
        )

    if transition_context.get("flattening_pre_close_residual") and not active_order_ids:
        return _result(
            snapshot,
            ReconciliationStatus.CONTROLLED_FLATTENING,
            Severity.WARNING,
            ("PROVEN_RESIDUAL_AWAITING_FLATTEN_CLOSE",),
            holding_policy=holding_policy,
            incident_required=False,
            auto_recoverable=True,
        )

    coverage = calculate_protective_coverage(position, snapshot.orders)
    codes = coverage.issue_codes
    if "PROTECTION_STOP_PRICE_INVALID" in codes:
        status = ReconciliationStatus.INCIDENT_MISMATCH
    elif "WRONG_SIDE_PROTECTION" in codes:
        status = ReconciliationStatus.INCIDENT_WRONG_SIDE
    elif coverage.valid_active_stop_quantity == 0:
        status = ReconciliationStatus.INCIDENT_UNPROTECTED
    elif coverage.uncovered_quantity > 0:
        status = ReconciliationStatus.INCIDENT_UNDERCOVERED
    elif coverage.overcovered_quantity > 0:
        status = ReconciliationStatus.INCIDENT_OVERCOVERED
    elif "MULTIPLE_ACTIVE_STOPS_IN_GROUP" in codes:
        status = ReconciliationStatus.INCIDENT_MISMATCH
    else:
        status = ReconciliationStatus.HEALTHY_PROTECTED

    if status == ReconciliationStatus.HEALTHY_PROTECTED:
        return _result(
            snapshot,
            status,
            Severity.INFO,
            codes,
            holding_policy=holding_policy,
            incident_required=False,
            auto_recoverable=True,
            coverage=(coverage,),
        )
    return _result(
        snapshot,
        status,
        Severity.CRITICAL,
        codes,
        holding_policy=holding_policy,
        incident_required=True,
        auto_recoverable=False,
        coverage=(coverage,),
        operator_action_required=True,
    )


def _incident_id(reason_codes: Sequence[str], snapshot_hash: str, observed_at: datetime) -> str:
    seed = "|".join(sorted(reason_codes)) + "|" + snapshot_hash + "|" + _iso(observed_at)
    return f"safety-{observed_at:%Y%m%d}-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def apply_safety_state_transition(
    current: Optional[SafetyState],
    result: ReconciliationResult,
    *,
    observed_at: datetime,
    incident_id_factory: Optional[Callable[[Sequence[str], str, datetime], str]] = None,
) -> SafetyState:
    """Apply deterministic transient/recovery/incident latch semantics."""

    state = current or SafetyState()
    if not isinstance(state, SafetyState):
        raise BrokerNormalizationError("current safety state must be a SafetyState")
    now = _datetime(observed_at, required=True)
    last_evidence_at = _datetime(state.last_evidence_at)
    if last_evidence_at is not None and now < last_evidence_at:
        raise ValueError("safety evidence timestamp predates the latest persisted evidence")
    is_unknown = result.status == ReconciliationStatus.BLOCKED_UNKNOWN_TRANSIENT
    is_healthy = result.status in HEALTHY_STATUSES and not result.incident_required
    incident = state.incident
    if (
        incident is not None
        and incident.state == SafetyIncidentState.CLEARED
        and incident.clear_metadata is not None
        and now > incident.clear_metadata.timestamp
    ):
        # CLEARED is historical, not a live latch. Once later evidence arrives,
        # its audited clear remains in the prior canonical/history event while
        # the current state starts a new evidence epoch.
        incident = None
    unknown_count = state.consecutive_unknown_count
    unknown_since = state.unknown_since
    recovery_hash = state.recovery_snapshot_hash
    recovery_count = state.consecutive_healthy_recovery_count
    transient_blocked = state.transient_blocked
    total_unknown_count = state.total_unknown_observation_count
    last_unknown_at = state.last_unknown_at
    last_unknown_hash = state.last_unknown_snapshot_hash
    last_unknown_codes = state.last_unknown_issue_codes

    if is_unknown:
        unknown_count += 1
        total_unknown_count += 1
        unknown_since = unknown_since or now
        last_unknown_at = now
        last_unknown_hash = result.snapshot_hash
        last_unknown_codes = result.issue_codes
        recovery_hash = None
        recovery_count = 0
        transient_blocked = True
    elif is_healthy:
        unknown_count = 0
        unknown_since = None
        if recovery_hash == result.snapshot_hash:
            recovery_count += 1
        else:
            recovery_hash = result.snapshot_hash
            recovery_count = 1
        if recovery_count >= REQUIRED_HEALTHY_RECOVERY_SNAPSHOTS:
            transient_blocked = False
    else:
        unknown_count = 0
        unknown_since = None
        recovery_hash = None
        recovery_count = 0
        transient_blocked = True

    persistent_codes: tuple[str, ...] = ()
    if result.incident_required:
        persistent_codes = result.issue_codes or (result.status.value,)
    elif is_unknown and unknown_since is not None:
        elapsed = (now - unknown_since).total_seconds()
        if (
            unknown_count >= MAX_CONSECUTIVE_UNKNOWN_SNAPSHOTS
            and elapsed >= MAX_UNKNOWN_ELAPSED_SECONDS
        ):
            persistent_codes = ("BROKER_STATE_UNKNOWN_PROLONGED",) + result.issue_codes

    if persistent_codes:
        if incident is None or incident.state == SafetyIncidentState.CLEARED:
            factory = incident_id_factory or _incident_id
            incident = SafetyIncident(
                incident_id=factory(persistent_codes, result.snapshot_hash, now),
                state=SafetyIncidentState.OPEN,
                reason_codes=tuple(sorted(set(persistent_codes))),
                first_observed=now,
                last_observed=now,
                occurrence_count=1,
                first_snapshot_hash=result.snapshot_hash,
                latest_snapshot_hash=result.snapshot_hash,
            )
        else:
            incident = replace(
                incident,
                state=SafetyIncidentState.OPEN,
                reason_codes=tuple(sorted(set(incident.reason_codes).union(persistent_codes))),
                last_observed=now,
                occurrence_count=incident.occurrence_count + 1,
                latest_snapshot_hash=result.snapshot_hash,
                clear_metadata=None,
            )
        transient_blocked = True
    elif incident is not None and incident.state in {SafetyIncidentState.OPEN, SafetyIncidentState.CLEARABLE}:
        if is_healthy and recovery_count >= REQUIRED_HEALTHY_RECOVERY_SNAPSHOTS:
            incident = replace(
                incident,
                state=SafetyIncidentState.CLEARABLE,
                last_observed=now,
                occurrence_count=incident.occurrence_count + 1,
                latest_snapshot_hash=result.snapshot_hash,
            )
        else:
            incident = replace(
                incident,
                state=SafetyIncidentState.OPEN,
                last_observed=now,
                occurrence_count=incident.occurrence_count + 1,
                latest_snapshot_hash=result.snapshot_hash,
                clear_metadata=None,
            )
        transient_blocked = True

    return SafetyState(
        transient_blocked=transient_blocked,
        consecutive_unknown_count=unknown_count,
        unknown_since=unknown_since,
        recovery_snapshot_hash=recovery_hash,
        consecutive_healthy_recovery_count=recovery_count,
        incident=incident,
        last_status=result.status,
        last_snapshot_hash=result.snapshot_hash,
        evidence_count=state.evidence_count + 1,
        total_unknown_observation_count=total_unknown_count,
        last_unknown_at=last_unknown_at,
        last_unknown_snapshot_hash=last_unknown_hash,
        last_unknown_issue_codes=last_unknown_codes,
        last_evidence_at=now,
    )


def can_submit_entry(
    result: ReconciliationResult,
    *,
    holding_policy: HoldingPolicy = DEFAULT_HOLDING_POLICY,
    safety_state: Optional[SafetyState] = None,
    pre_submit_snapshot_hash: Optional[str] = None,
    policy_authorized: bool = False,
) -> bool:
    """Authorize entry only from complete, explicit, independently supplied evidence."""

    try:
        resolved_policy = HoldingPolicy(holding_policy)
    except (TypeError, ValueError):
        return False
    if resolved_policy == HoldingPolicy.UNRESOLVED or policy_authorized is not True:
        return False
    if (
        result.status != ReconciliationStatus.HEALTHY_FLAT
        or result.incident_required is not False
        or result.entry_allowed is not False
    ):
        return False
    if not isinstance(safety_state, SafetyState):
        return False
    try:
        validated_state = safety_state_from_dict(safety_state_to_dict(safety_state))
    except (AttributeError, BrokerNormalizationError, TypeError, ValueError):
        return False
    if validated_state != safety_state:
        return False
    if (
        safety_state.evidence_count <= 0
        or safety_state.last_status != ReconciliationStatus.HEALTHY_FLAT
        or safety_state.last_snapshot_hash != result.snapshot_hash
        or safety_state.last_evidence_at is None
    ):
        return False
    if safety_state.transient_blocked is not False:
        return False
    if safety_state.consecutive_unknown_count != 0 or safety_state.unknown_since is not None:
        return False
    if (
        safety_state.incident is not None
        and safety_state.incident.state in {SafetyIncidentState.OPEN, SafetyIncidentState.CLEARABLE}
    ):
        return False
    if pre_submit_snapshot_hash is None or pre_submit_snapshot_hash != result.snapshot_hash:
        return False
    return True


def _validate_safety_state_timestamps(state: SafetyState) -> None:
    """Validate persisted evidence ordering without changing the supplied state."""

    last_evidence_at = _datetime(state.last_evidence_at)
    unknown_since = _datetime(state.unknown_since)
    last_unknown_at = _datetime(state.last_unknown_at)
    if last_evidence_at is not None:
        if unknown_since is not None and unknown_since > last_evidence_at:
            raise BrokerNormalizationError("unknown_since is later than last_evidence_at")
        if last_unknown_at is not None and last_unknown_at > last_evidence_at:
            raise BrokerNormalizationError("last_unknown_at is later than last_evidence_at")
    incident = state.incident
    if incident is None:
        return
    first_observed = _datetime(incident.first_observed, required=True)
    last_observed = _datetime(incident.last_observed, required=True)
    if first_observed > last_observed:
        raise BrokerNormalizationError("incident observation timestamps are inverted")
    if last_evidence_at is not None and last_observed > last_evidence_at:
        raise BrokerNormalizationError("incident last_observed is later than last_evidence_at")
    if incident.clear_metadata is not None:
        cleared_at = _datetime(incident.clear_metadata.timestamp, required=True)
        if (
            cleared_at < first_observed
            or cleared_at < last_observed
            or (last_evidence_at is not None and cleared_at < last_evidence_at)
        ):
            raise BrokerNormalizationError("incident clear timestamp predates incident evidence")


def safety_state_to_dict(state: SafetyState) -> dict[str, Any]:
    if not isinstance(state, SafetyState):
        raise BrokerNormalizationError("safety state must be a SafetyState")
    _validate_safety_state_timestamps(state)
    incident = state.incident
    incident_payload: Optional[dict[str, Any]] = None
    if incident is not None:
        clear = incident.clear_metadata
        incident_payload = {
            "incident_id": incident.incident_id,
            "state": incident.state.value,
            "reason_codes": list(incident.reason_codes),
            "first_observed": _iso(incident.first_observed),
            "last_observed": _iso(incident.last_observed),
            "occurrence_count": incident.occurrence_count,
            "first_snapshot_hash": incident.first_snapshot_hash,
            "latest_snapshot_hash": incident.latest_snapshot_hash,
            "clear_metadata": None
            if clear is None
            else {
                "incident_id": clear.incident_id,
                "operator": clear.operator,
                "reason": clear.reason,
                "timestamp": _iso(clear.timestamp),
                "verification_snapshot_hashes": list(clear.verification_snapshot_hashes),
            },
        }
    payload = {
        "schema_version": 1,
        "transient_blocked": state.transient_blocked,
        "consecutive_unknown_count": state.consecutive_unknown_count,
        "unknown_since": _iso(state.unknown_since),
        "recovery_snapshot_hash": state.recovery_snapshot_hash,
        "consecutive_healthy_recovery_count": state.consecutive_healthy_recovery_count,
        "incident": incident_payload,
        "last_status": None if state.last_status is None else state.last_status.value,
        "last_snapshot_hash": state.last_snapshot_hash,
        "evidence_count": state.evidence_count,
        "total_unknown_observation_count": state.total_unknown_observation_count,
        "last_unknown_at": _iso(state.last_unknown_at),
        "last_unknown_snapshot_hash": state.last_unknown_snapshot_hash,
        "last_unknown_issue_codes": list(state.last_unknown_issue_codes),
        "last_evidence_at": _iso(state.last_evidence_at),
    }
    validated = safety_state_from_dict(payload)
    if validated != state:
        raise BrokerNormalizationError("safety state is not in canonical persisted form")
    return payload


def safety_state_from_dict(payload: Optional[Mapping[str, Any]]) -> SafetyState:
    if payload is None:
        return SafetyState()
    if not isinstance(payload, Mapping):
        raise BrokerNormalizationError("persisted safety state must be a JSON object")
    if not payload:
        raise BrokerNormalizationError("persisted safety state must not be empty")
    if payload.get("schema_version") != 1:
        raise BrokerNormalizationError("unsupported persisted safety-state schema")

    def strict_bool(name: str, default: bool = False) -> bool:
        value = payload.get(name, default)
        if type(value) is not bool:
            raise BrokerNormalizationError(f"persisted {name} must be boolean")
        return value

    def nonnegative_int(name: str, default: int = 0) -> int:
        value = payload.get(name, default)
        if type(value) is not int or value < 0:
            raise BrokerNormalizationError(f"persisted {name} must be a nonnegative integer")
        return value

    unknown_count = nonnegative_int("consecutive_unknown_count")
    recovery_count = nonnegative_int("consecutive_healthy_recovery_count")
    evidence_count = nonnegative_int("evidence_count")
    total_unknown_count = nonnegative_int("total_unknown_observation_count")
    unknown_since = _datetime(payload.get("unknown_since"))
    last_unknown_at = _datetime(payload.get("last_unknown_at"))
    last_evidence_at = _datetime(payload.get("last_evidence_at"))
    recovery_hash = _optional_text(payload.get("recovery_snapshot_hash"))
    last_unknown_hash = _optional_text(payload.get("last_unknown_snapshot_hash"))
    last_unknown_codes_raw = payload.get("last_unknown_issue_codes", ())
    if not isinstance(last_unknown_codes_raw, (list, tuple)):
        raise BrokerNormalizationError("persisted last unknown issue codes must be a list")
    last_unknown_codes = tuple(
        _required_text(code, "last unknown issue code") for code in last_unknown_codes_raw
    )
    if unknown_count > 0 and unknown_since is None:
        raise BrokerNormalizationError("persisted unknown count lacks unknown_since evidence")
    if recovery_count > 0 and recovery_hash is None:
        raise BrokerNormalizationError("persisted recovery count lacks a semantic snapshot hash")
    if total_unknown_count > 0 and (last_unknown_at is None or last_unknown_hash is None):
        raise BrokerNormalizationError("persisted unknown history lacks its latest evidence")
    if total_unknown_count > 0 and not last_unknown_codes:
        raise BrokerNormalizationError("persisted unknown history lacks issue-code evidence")

    incident_payload = payload.get("incident")
    incident: Optional[SafetyIncident] = None
    if incident_payload is not None:
        if not isinstance(incident_payload, Mapping):
            raise BrokerNormalizationError("persisted safety incident must be a JSON object")
        if not incident_payload:
            raise BrokerNormalizationError("persisted safety incident must not be empty")
        clear_payload = incident_payload.get("clear_metadata")
        clear = None
        if clear_payload is not None:
            if not isinstance(clear_payload, Mapping):
                raise BrokerNormalizationError("persisted clear metadata must be a JSON object")
            if not clear_payload:
                raise BrokerNormalizationError("persisted clear metadata must not be empty")
            verification_hashes_raw = clear_payload.get("verification_snapshot_hashes", ())
            if not isinstance(verification_hashes_raw, (list, tuple)):
                raise BrokerNormalizationError(
                    "persisted clear verification hashes must be a list"
                )
            verification_hashes = tuple(verification_hashes_raw)
            if (
                len(verification_hashes) < REQUIRED_HEALTHY_RECOVERY_SNAPSHOTS
                or len(set(verification_hashes)) != 1
                or any(not _optional_text(value) for value in verification_hashes)
            ):
                raise BrokerNormalizationError("persisted clear verification hashes are invalid")
            clear = IncidentClearMetadata(
                incident_id=_required_text(clear_payload.get("incident_id"), "clear incident id"),
                operator=_required_text(clear_payload.get("operator"), "clear operator"),
                reason=_required_text(clear_payload.get("reason"), "clear reason"),
                timestamp=_datetime(clear_payload.get("timestamp"), required=True),
                verification_snapshot_hashes=verification_hashes,
            )
        reason_codes_raw = incident_payload.get("reason_codes", ())
        if not isinstance(reason_codes_raw, (list, tuple)):
            raise BrokerNormalizationError("persisted incident reason codes must be a list")
        reason_codes = tuple(_required_text(code, "incident reason code") for code in reason_codes_raw)
        if not reason_codes:
            raise BrokerNormalizationError("persisted incident requires reason codes")
        occurrence_count = incident_payload.get("occurrence_count", 0)
        if type(occurrence_count) is not int or occurrence_count < 1:
            raise BrokerNormalizationError("persisted incident occurrence count must be positive")
        incident_state = SafetyIncidentState(incident_payload.get("state"))
        if incident_state == SafetyIncidentState.CLEARED and clear is None:
            raise BrokerNormalizationError("CLEARED incident lacks clear metadata")
        if incident_state != SafetyIncidentState.CLEARED and clear is not None:
            raise BrokerNormalizationError("uncleared incident contains clear metadata")
        incident_id = _required_text(incident_payload.get("incident_id"), "incident id")
        if clear is not None and clear.incident_id != incident_id:
            raise BrokerNormalizationError("clear metadata incident ID does not match incident")
        first_observed = _datetime(incident_payload.get("first_observed"), required=True)
        last_observed = _datetime(incident_payload.get("last_observed"), required=True)
        if first_observed > last_observed:
            raise BrokerNormalizationError("incident observation timestamps are inverted")
        if clear is not None and (
            clear.timestamp < first_observed or clear.timestamp < last_observed
        ):
            raise BrokerNormalizationError("incident clear timestamp predates incident evidence")
        incident = SafetyIncident(
            incident_id=incident_id,
            state=incident_state,
            reason_codes=reason_codes,
            first_observed=first_observed,
            last_observed=last_observed,
            occurrence_count=occurrence_count,
            first_snapshot_hash=_required_text(
                incident_payload.get("first_snapshot_hash"), "first snapshot hash"
            ),
            latest_snapshot_hash=_required_text(
                incident_payload.get("latest_snapshot_hash"), "latest snapshot hash"
            ),
            clear_metadata=clear,
        )
    last_status = payload.get("last_status")
    transient_blocked = strict_bool("transient_blocked")
    if unknown_count > 0 and not transient_blocked:
        raise BrokerNormalizationError("persisted unknown observations are not fail-closed")
    if (
        incident is not None
        and incident.state in {SafetyIncidentState.OPEN, SafetyIncidentState.CLEARABLE}
        and not transient_blocked
    ):
        raise BrokerNormalizationError("persisted uncleared incident is not fail-closed")
    if incident is not None and incident.state == SafetyIncidentState.CLEARABLE:
        if (
            recovery_count < REQUIRED_HEALTHY_RECOVERY_SNAPSHOTS
            or recovery_hash != incident.latest_snapshot_hash
        ):
            raise BrokerNormalizationError("CLEARABLE incident lacks matching recovery evidence")
    if incident is not None and incident.state == SafetyIncidentState.CLEARED:
        hashes = incident.clear_metadata.verification_snapshot_hashes
        if hashes[0] != incident.latest_snapshot_hash:
            raise BrokerNormalizationError("clear verification does not match latest incident snapshot")
    state = SafetyState(
        transient_blocked=transient_blocked,
        consecutive_unknown_count=unknown_count,
        unknown_since=unknown_since,
        recovery_snapshot_hash=recovery_hash,
        consecutive_healthy_recovery_count=recovery_count,
        incident=incident,
        last_status=None if not last_status else ReconciliationStatus(last_status),
        last_snapshot_hash=_optional_text(payload.get("last_snapshot_hash")),
        evidence_count=evidence_count,
        total_unknown_observation_count=total_unknown_count,
        last_unknown_at=last_unknown_at,
        last_unknown_snapshot_hash=last_unknown_hash,
        last_unknown_issue_codes=last_unknown_codes,
        last_evidence_at=last_evidence_at,
    )
    _validate_safety_state_timestamps(state)
    return state


def clear_safety_incident(
    state: SafetyState,
    *,
    incident_id: str,
    operator: str,
    reason: str,
    verification_snapshot_hashes: Sequence[str],
    cleared_at: datetime,
) -> SafetyState:
    """Explicitly clear a CLEARABLE incident with auditable verification facts."""

    if state.incident is None or state.incident.state != SafetyIncidentState.CLEARABLE:
        raise ValueError("no CLEARABLE safety incident exists")
    if incident_id != state.incident.incident_id:
        raise ValueError("incident ID does not match the latched incident")
    operator = _required_text(operator, "operator")
    reason = _required_text(reason, "clear reason")
    hashes = tuple(_required_text(value, "verification snapshot hash") for value in verification_snapshot_hashes)
    if len(hashes) < REQUIRED_HEALTHY_RECOVERY_SNAPSHOTS or len(set(hashes)) != 1:
        raise ValueError("clear requires two semantically equivalent verification snapshots")
    if (
        state.consecutive_healthy_recovery_count < REQUIRED_HEALTHY_RECOVERY_SNAPSHOTS
        or state.recovery_snapshot_hash is None
        or hashes[0] != state.recovery_snapshot_hash
        or hashes[0] != state.incident.latest_snapshot_hash
    ):
        raise ValueError("verification hashes do not match the clearable healthy broker state")
    cleared = _datetime(cleared_at, required=True)
    if (
        cleared < state.incident.first_observed
        or cleared < state.incident.last_observed
        or (state.last_evidence_at is not None and cleared < state.last_evidence_at)
    ):
        raise ValueError("clear timestamp predates incident or healthy recovery evidence")
    metadata = IncidentClearMetadata(
        incident_id=incident_id,
        operator=operator,
        reason=reason,
        timestamp=cleared,
        verification_snapshot_hashes=hashes,
    )
    candidate = replace(
        state,
        transient_blocked=False,
        incident=replace(
            state.incident,
            state=SafetyIncidentState.CLEARED,
            clear_metadata=metadata,
        ),
    )
    return safety_state_from_dict(safety_state_to_dict(candidate))
