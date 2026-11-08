"""Deterministic, profile-aware feature generation for the VWAP parity work.

This module is intentionally standalone in Stage 1A.  Existing live, replay,
and audit code does not import it yet, so creating it cannot change active
trading behaviour.  The profiles document the distinct historical VWAP bases
that will be wired into those paths in a later, reviewed stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

try:
    import pandas_market_calendars as mcal
except ModuleNotFoundError:  # pragma: no cover - exercised by dependency-absence tests.
    mcal = None


REQUIRED_BAR_COLUMNS = frozenset({"timestamp", "open", "high", "low", "close", "volume"})
NY_TZ = "America/New_York"
NYSE = mcal.get_calendar("NYSE") if mcal is not None else None


@dataclass(frozen=True)
class FeatureProfile:
    """A named, explicit VWAP reproducibility contract."""

    name: str
    vwap_mode: str
    vwap_window: str
    session_scoped: bool


LEGACY_LIVE_7CALENDAR_CUMULATIVE_VWAP_V1 = FeatureProfile(
    name="legacy_live_7calendar_cumulative_vwap_v1",
    vwap_mode="cumulative_close_x_volume",
    vwap_window="fetched_7_calendar_days",
    session_scoped=False,
)

LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1 = FeatureProfile(
    name="legacy_replay_full_csv_cumulative_vwap_v1",
    vwap_mode="cumulative_close_x_volume",
    vwap_window="full_supplied_csv",
    session_scoped=False,
)

SESSION_REGULAR_NY_VWAP_V1 = FeatureProfile(
    name="session_regular_ny_vwap_v1",
    vwap_mode="session_regular_ny_close_x_volume",
    vwap_window="nyse_regular_session",
    session_scoped=True,
)

FEATURE_PROFILES: Mapping[str, FeatureProfile] = {
    profile.name: profile
    for profile in (
        LEGACY_LIVE_7CALENDAR_CUMULATIVE_VWAP_V1,
        LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1,
        SESSION_REGULAR_NY_VWAP_V1,
    )
}


@dataclass(frozen=True)
class SignalDiagnostic:
    """Diagnostic mirror of the current raw mean-reversion signal gates."""

    raw_signal: int
    final_signal_veto_reason: str


def get_feature_profile(profile: str | FeatureProfile) -> FeatureProfile:
    """Resolve a supported feature profile by name without accepting aliases."""
    if isinstance(profile, FeatureProfile):
        if profile.name not in FEATURE_PROFILES:
            raise ValueError(f"Unsupported feature profile: {profile.name}")
        return profile
    try:
        return FEATURE_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported feature profile: {profile}") from exc


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic timestamp-ordered bar frame.

    Timestamps are parsed as UTC.  A stable sort followed by ``keep='last'``
    makes duplicate resolution explicit: the final source row for a timestamp
    wins.  Non-positive volume is retained for price-based indicators but is
    explicitly ineligible for VWAP accumulation.
    """
    missing = REQUIRED_BAR_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required bar columns: {sorted(missing)}")

    bars = df.copy()
    bars["_source_order"] = np.arange(len(bars))
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["timestamp"])

    for column in ("open", "high", "low", "close", "volume"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")

    bars = (
        bars.sort_values(["timestamp", "_source_order"], kind="mergesort")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .drop(columns=["_source_order"])
        .reset_index(drop=True)
    )
    bars["positive_volume_eligible"] = bars["volume"].gt(0).fillna(False)
    return bars


def _nyse_session_keys(timestamps: pd.Series) -> pd.Series:
    """Map timestamps inside an NYSE regular session to its NY session date."""
    keys = pd.Series(pd.NA, index=timestamps.index, dtype="string")
    valid = timestamps.notna()
    if not valid.any():
        return keys

    valid_ts = pd.DatetimeIndex(timestamps.loc[valid])
    start_date = valid_ts.min().tz_convert(NY_TZ).date()
    end_date = valid_ts.max().tz_convert(NY_TZ).date()
    schedule = _nyse_schedule(start_date, end_date)
    if schedule.empty:
        return keys

    opens = pd.DatetimeIndex(schedule["market_open"])
    closes = pd.DatetimeIndex(schedule["market_close"])
    positions = opens.searchsorted(valid_ts, side="right") - 1
    session_labels = pd.Series(schedule.index.date, index=np.arange(len(schedule))).astype(str)

    resolved: list[object] = []
    for timestamp, position in zip(valid_ts, positions):
        if position >= 0 and timestamp < closes[position]:
            resolved.append(session_labels.iloc[position])
        else:
            resolved.append(pd.NA)

    keys.loc[valid] = pd.array(resolved, dtype="string")
    return keys


def _nyse_schedule(start_date, end_date) -> pd.DataFrame:
    """Return NYSE regular-session bounds from the exchange calendar only."""
    if mcal is None or NYSE is None:
        raise RuntimeError(
            "session_regular_ny_vwap_v1 requires pandas_market_calendars "
            "with the NYSE exchange calendar; install pandas_market_calendars."
        )
    return NYSE.schedule(start_date=start_date, end_date=end_date)


def _cumulative_vwap(
    close: pd.Series,
    volume: pd.Series,
    eligible: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    eligible = eligible.fillna(False).astype(bool)
    contributing_volume = volume.where(eligible, 0.0).fillna(0.0)
    contributing_notional = (close * volume).where(eligible, 0.0).fillna(0.0)
    denominator = contributing_volume.cumsum()
    vwap = contributing_notional.cumsum().div(denominator.where(denominator > 0))
    included = eligible.astype(int).cumsum().astype(int)
    return vwap, included


def _session_vwap(
    close: pd.Series,
    volume: pd.Series,
    eligible: pd.Series,
    session_keys: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    in_session = session_keys.notna()
    contributing = eligible.fillna(False).astype(bool) & in_session
    vwap = pd.Series(np.nan, index=close.index, dtype=float)
    included = pd.Series(0, index=close.index, dtype=int)
    if not in_session.any():
        return vwap, included

    contributing_volume = volume.where(contributing, 0.0).fillna(0.0)
    contributing_notional = (close * volume).where(contributing, 0.0).fillna(0.0)
    groups = session_keys.loc[in_session]
    denominator = contributing_volume.loc[in_session].groupby(groups, sort=False).cumsum()
    numerator = contributing_notional.loc[in_session].groupby(groups, sort=False).cumsum()
    vwap.loc[in_session] = numerator.div(denominator.where(denominator > 0))
    included.loc[in_session] = (
        contributing.loc[in_session].astype(int).groupby(groups, sort=False).cumsum().astype(int)
    )
    return vwap, included


def evaluate_raw_signal(row: pd.Series) -> SignalDiagnostic:
    """Mirror the existing frozen raw-signal logic without modifying it.

    The thresholds intentionally match the current strategy: ATR must be in
    its percentile band; a long requires trend=1, VWAP distance below -0.8,
    and RSI below 30; a short is the symmetric trend=-1 / +0.8 / RSI>70 case.
    This helper is diagnostic-only until a later reviewed integration stage.
    """
    atr = row.get("atr", np.nan)
    vwap = row.get("vwap_used", np.nan)
    rsi = row.get("rsi", np.nan)
    trend15 = row.get("trend15", np.nan)
    band_low = row.get("atr_q_low_band", np.nan)
    band_high = row.get("atr_q_high_band", np.nan)

    if pd.isna(atr) or atr <= 0:
        return SignalDiagnostic(0, "INSUFFICIENT_ATR")
    if pd.isna(vwap) or pd.isna(rsi):
        return SignalDiagnostic(0, "INSUFFICIENT_VWAP_OR_RSI")
    if pd.isna(trend15) or trend15 == 0:
        return SignalDiagnostic(0, "INSUFFICIENT_TREND15")
    if pd.isna(band_low) or pd.isna(band_high):
        return SignalDiagnostic(0, "INSUFFICIENT_ATR_BAND_HISTORY")
    if not (atr >= band_low and atr <= band_high):
        return SignalDiagnostic(0, "ATR_OUT_OF_BAND")

    vwap_dist = row.get("vwap_dist", np.nan)
    if pd.isna(vwap_dist):
        return SignalDiagnostic(0, "INSUFFICIENT_VWAP_DISTANCE")
    if trend15 == 1 and vwap_dist < -0.8 and rsi < 30:
        return SignalDiagnostic(1, "")
    if trend15 == -1 and vwap_dist > 0.8 and rsi > 70:
        return SignalDiagnostic(-1, "")
    return SignalDiagnostic(0, "NO_VWAP_RSI_TREND_SETUP")


def generate_features(
    df: pd.DataFrame,
    profile: str | FeatureProfile,
    *,
    atr_window: int = 200,
    atr_q_low: float = 0.55,
    atr_q_high: float = 0.95,
    trend_fast: int = 21,
    trend_slow: int = 50,
    include_signal_diagnostics: bool = True,
    normalize: bool = True,
) -> pd.DataFrame:
    """Build deterministic strategy features under an explicit VWAP profile.

    ``normalize=False`` is for callers that have already applied a frozen input
    ordering/filtering contract and need to retain every row exactly as supplied.
    """
    if atr_window <= 0:
        raise ValueError("atr_window must be positive")
    resolved_profile = get_feature_profile(profile)
    if normalize:
        bars = normalize_bars(df)
    else:
        missing = REQUIRED_BAR_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required bar columns: {sorted(missing)}")
        bars = df.copy()
        bars["positive_volume_eligible"] = bars["volume"].gt(0).fillna(False)

    bars["feature_profile"] = resolved_profile.name
    bars["vwap_mode"] = resolved_profile.vwap_mode
    bars["vwap_window"] = resolved_profile.vwap_window

    if resolved_profile.session_scoped:
        session_keys = _nyse_session_keys(bars["timestamp"])
        bars["vwap_session_key"] = session_keys
        vwap, included = _session_vwap(
            bars["close"], bars["volume"], bars["positive_volume_eligible"], session_keys
        )
    else:
        bars["vwap_session_key"] = pd.Series(pd.NA, index=bars.index, dtype="string")
        vwap, included = _cumulative_vwap(
            bars["close"], bars["volume"], bars["positive_volume_eligible"]
        )

    bars["vwap_used"] = vwap
    bars["included_bar_count"] = included

    delta = bars["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    bars["rsi"] = 100 - (100 / (1 + rs))

    tr = np.maximum(
        bars["high"] - bars["low"],
        np.maximum(
            (bars["high"] - bars["close"].shift()).abs(),
            (bars["low"] - bars["close"].shift()).abs(),
        ),
    )
    bars["atr"] = tr.rolling(14).mean()
    bars["atr_q_low_band"] = bars["atr"].rolling(atr_window).quantile(atr_q_low)
    bars["atr_q_high_band"] = bars["atr"].rolling(atr_window).quantile(atr_q_high)
    bars["atr_band_result"] = np.select(
        [
            bars["atr"].isna() | bars["atr_q_low_band"].isna() | bars["atr_q_high_band"].isna(),
            (bars["atr"] >= bars["atr_q_low_band"]) & (bars["atr"] <= bars["atr_q_high_band"]),
        ],
        ["UNAVAILABLE", "IN_BAND"],
        default="OUT_OF_BAND",
    )

    indexed = bars.set_index("timestamp")
    bars_15m = (
        indexed.resample("15min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )
    bars_15m["ema_fast"] = bars_15m["close"].ewm(span=trend_fast, adjust=False).mean()
    bars_15m["ema_slow"] = bars_15m["close"].ewm(span=trend_slow, adjust=False).mean()
    bars_15m["trend15"] = np.where(bars_15m["ema_fast"] > bars_15m["ema_slow"], 1, -1)
    bars_15m["trend15_lagged"] = bars_15m["trend15"].shift(1)
    bars["trend15"] = (
        bars_15m["trend15_lagged"].reindex(indexed.index, method="ffill").fillna(0).astype(int).values
    )

    bars["vwap_dist"] = (bars["close"] - bars["vwap_used"]) / bars["atr"]
    if include_signal_diagnostics:
        diagnostics = bars.apply(evaluate_raw_signal, axis=1)
        bars["raw_signal"] = [diagnostic.raw_signal for diagnostic in diagnostics]
        bars["final_signal_veto_reason"] = [
            diagnostic.final_signal_veto_reason for diagnostic in diagnostics
        ]
    return bars
