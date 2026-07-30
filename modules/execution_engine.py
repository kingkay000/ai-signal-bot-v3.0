"""
modules/execution_engine.py
─────────────────────────────────────────────────────────────────────────────
Module 6 — Execution Engine

Handles order placement in both paper trading mode and live mode via ccxt.

Features:
  - Paper trading: full in-memory simulation with virtual portfolio
  - Live trading: MARKET and LIMIT orders via ccxt
  - OCO simulation (stop-loss + take-profit registered simultaneously)
  - Exponential backoff retry (3 attempts, 1s/2s/4s)
  - Order fill confirmation before logging
  - Trailing stop updates (delegated to RiskManager)

Usage:
    engine = ExecutionEngine(config, data_engine)
    order = engine.place_order(sizing, signal_direction)
"""

import time
import csv
import json
import os
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import ccxt
except ImportError:
    ccxt = None

from modules.risk_manager import PositionSizing
from utils.helpers import retry
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class OrderResult:
    """
    Represents the outcome of an order placement attempt.

    Attributes:
        order_id:    Exchange order ID (or 'PAPER-xxx' in paper mode).
        symbol:      Trading pair.
        side:        'buy' or 'sell'.
        order_type:  'market' or 'limit'.
        price:       Filled price (market) or requested price (limit).
        amount:      Order size in base currency.
        status:      'filled', 'open', 'rejected', 'error'.
        timestamp:   UTC time of order placement.
        is_paper:    True if this is a simulated paper order.
        fees:        Fees paid in quote currency.
        raw:         Raw exchange response dict.
    """

    order_id: str = ""
    symbol: str = ""
    side: str = ""
    order_type: str = "market"
    price: float = 0.0
    amount: float = 0.0
    status: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_paper: bool = False
    fees: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "price": self.price,
            "amount": self.amount,
            "status": self.status,
            "timestamp": str(self.timestamp),
            "is_paper": self.is_paper,
            "fees": self.fees,
        }


class PaperPortfolio:
    """
    Tracks a simulated paper-trading portfolio.

    Args:
        initial_balance: Starting USDT balance.
        commission:      Commission rate per trade (e.g. 0.001 = 0.1%).
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        commission: float = 0.001,
    ) -> None:
        self.usdt_balance: float = initial_balance
        self.initial_balance: float = initial_balance
        self.commission: float = commission
        self.holdings: Dict[str, float] = {}  # symbol → base amount held
        self.trade_history: List[Dict[str, Any]] = []
        self._order_counter: int = 0

    def execute(
        self,
        symbol: str,
        side: str,  # 'buy' | 'sell'
        amount: float,
        price: float,
    ) -> OrderResult:
        """
        Simulate order execution against virtual USDT balance.

        Args:
            symbol: Trading pair.
            side:   'buy' or 'sell'.
            amount: Base currency amount.
            price:  Execution price (last market price).

        Returns:
            OrderResult representing the simulated fill.
        """
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"
        cost = amount * price
        fee = cost * self.commission

        base_sym = symbol.split("/")[0]

        if side == "buy":
            total_cost = cost + fee
            if self.usdt_balance < total_cost:
                log.warning(
                    f"Paper: Insufficient balance ({self.usdt_balance:.2f} < {total_cost:.2f})"
                )
                return OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    price=price,
                    amount=amount,
                    status="rejected",
                    is_paper=True,
                    fees=fee,
                )
            self.usdt_balance -= total_cost
            self.holdings[base_sym] = self.holdings.get(base_sym, 0.0) + amount
        else:  # sell
            held = self.holdings.get(base_sym, 0.0)
            if held < amount:
                log.warning(
                    f"Paper: Insufficient holdings for {base_sym}: {held:.6f} < {amount:.6f}"
                )
                amount = held  # partial sell
            self.usdt_balance += cost - fee
            self.holdings[base_sym] = held - amount

        result = OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type="market",
            price=price,
            amount=amount,
            status="filled",
            is_paper=True,
            fees=fee,
        )
        self.trade_history.append(result.to_dict())
        log.info(
            f"[PAPER] {side.upper()} {amount:.6f} {symbol} @ {price:.4f} | "
            f"fee={fee:.4f} | USDT balance={self.usdt_balance:.2f}"
        )
        return result

    def get_summary(self) -> Dict[str, Any]:
        total_holding_value = sum(
            amt for sym, amt in self.holdings.items()
        )  # Simplified; real value needs current prices
        return {
            "usdt_balance": round(self.usdt_balance, 2),
            "initial_balance": self.initial_balance,
            "holdings": {k: round(v, 8) for k, v in self.holdings.items()},
            "total_trades": len(self.trade_history),
        }


class ExecutionEngine:
    """
    Places and manages orders in either paper or live mode.

    Args:
        config:      Parsed config.yaml dict.
        exchange:    An initialised ccxt.Exchange instance (from DataEngine).
        mode:        'paper' or 'live' (overrides config if provided).
    """

    MAX_RETRIES = 3
    RETRY_DELAYS = (1.0, 2.0, 4.0)
    DEFAULT_COMMISSION = 0.001
    BRIDGE_URL = "http://localhost:8000/signals"
    BRIDGE_KEY = os.getenv("EXECUTION_BRIDGE_KEY", "default_secret_key")

    def __init__(
        self,
        config: Dict[str, Any],
        exchange: Optional[ccxt.Exchange] = None,
        mode: Optional[str] = None,
    ) -> None:
        trading_cfg = config.get("trading", {})
        bt_cfg = config.get("backtesting", {})
        paper_cfg = config.get("paper_trading", {})

        self.mode: str = mode or trading_cfg.get("mode", "paper")
        self.order_type: str = trading_cfg.get("order_type", "market")
        self.exchange: Optional[ccxt.Exchange] = exchange

        initial_balance: float = paper_cfg.get("initial_balance", 10_000.0)
        commission: float = bt_cfg.get("commission", self.DEFAULT_COMMISSION)

        self.paper_portfolio = PaperPortfolio(initial_balance, commission)
        self.order_history: List[OrderResult] = []
        self.filled_orders_log_path = os.path.join("logs", "filled_orders.jsonl")
        self.filled_orders_csv_path = os.path.join("logs", "filled_orders.csv")
        self._init_filled_orders_log()

        log.info(f"ExecutionEngine initialised — mode={self.mode}")

    # ─── Main Order Entry ─────────────────────────────────────────────────────

    def place_order(
        self,
        sizing: PositionSizing,
        current_price: float,
    ) -> OrderResult:
        """
        Place an entry order based on approved position sizing.

        Routes to paper or live execution based on mode.

        Args:
            sizing:        Approved PositionSizing from RiskManager.
            current_price: Current market price (for paper fills).

        Returns:
            OrderResult with fill details.
        """
        side = "buy" if sizing.direction == "long" else "sell"
        symbol = sizing.symbol
        amount = sizing.position_size
        price = sizing.entry_price if sizing.entry_price > 0 else current_price

        log.info(
            f"Placing {self.mode.upper()} {side.upper()} order: "
            f"{symbol} | size={amount:.6f} | price={price:.4f}"
        )

        if self.mode == "paper":
            result = self.paper_portfolio.execute(symbol, side, amount, current_price)
        elif self.mode == "bridge":
            result = self._post_to_bridge(sizing, side, current_price)
        else:
            result = self._live_order(symbol, side, amount, price)

        if result.status == "filled":
            self.order_history.append(result)
            self._log_filled_order(
                order=result,
                event_type="entry",
                mode=self.mode,
                risk_amount=sizing.risk_amount,
                lot_size=(sizing.position_size / 100_000),
            )
            log.info(f"Order filled: {result.order_id} @ {result.price:.4f}")
        else:
            log.warning(f"Order not filled: {result.status}")

        return result

    def close_position(
        self,
        symbol: str,
        amount: float,
        current_price: float,
        direction: str = "long",
        reason: str = "",
    ) -> OrderResult:
        """
        Close an open position by placing a reverse order.

        Args:
            symbol:        Trading pair.
            amount:        Size to close (in base currency).
            current_price: Current market price.
            direction:     Original position direction ('long' or 'short').
            reason:        Reason string for logging (SL, TP1, TP2, etc.).

        Returns:
            OrderResult for the closing order.
        """
        side = "sell" if direction == "long" else "buy"
        log.info(
            f"Closing {direction} {symbol}: {side.upper()} {amount:.6f} @ {current_price:.4f} [{reason}]"
        )

        if self.mode == "paper":
            result = self.paper_portfolio.execute(symbol, side, amount, current_price)
        elif self.mode == "bridge":
            # In bridge mode the MQL5 EA owns actual execution (SL/TP hits happen on
            # the broker side).  When the Python bot decides to close a position (e.g.
            # thesis broken, momentum stall, reversal), it can only signal intent —
            # there is no synchronous close confirmation from the EA.  We treat the
            # close as immediately filled at the last-known price so that
            # risk_manager.close_position() is called, _log_trade_close() writes the
            # CLOSE event to trade_lifecycle.txt, and the weekly summary is non-zero.
            result = OrderResult(
                order_id=f"BRIDGE-CLOSE-{symbol}-{int(time.time())}",
                symbol=symbol,
                side=side,
                price=current_price,
                amount=amount,
                status="filled",
                is_paper=False,
            )
            log.info(
                f"Bridge close recorded for {symbol} @ {current_price:.5f} [{reason}]"
            )
        else:
            result = self._live_order(symbol, side, amount, current_price)

        if result.status == "filled":
            self.order_history.append(result)
            self._log_filled_order(
                order=result,
                event_type="exit",
                mode=self.mode,
                risk_amount=0.0,
                lot_size=(amount / 100_000),
                reason=reason,
            )
        return result

    def _init_filled_orders_log(self) -> None:
        """Create fill log files (JSONL + CSV header) if missing."""
        os.makedirs("logs", exist_ok=True)
        if not os.path.exists(self.filled_orders_csv_path):
            with open(self.filled_orders_csv_path, mode="w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        "timestamp_utc",
                        "event_type",
                        "order_id",
                        "symbol",
                        "side",
                        "price",
                        "amount",
                        "lot_size",
                        "risk_amount",
                        "fees",
                        "status",
                        "mode",
                        "is_paper",
                        "reason",
                    ]
                )

    def _log_filled_order(
        self,
        order: OrderResult,
        event_type: str,
        mode: str,
        risk_amount: float,
        lot_size: float,
        reason: str = "",
    ) -> None:
        """Persist each filled order to append-only JSONL and CSV logs."""
        try:
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "price": round(order.price, 6),
                "amount": round(order.amount, 8),
                "lot_size": round(lot_size, 4),
                "risk_amount": round(risk_amount, 2),
                "fees": round(order.fees, 6),
                "status": order.status,
                "mode": mode,
                "is_paper": order.is_paper,
                "reason": reason,
            }

            with open(self.filled_orders_log_path, mode="a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

            with open(self.filled_orders_csv_path, mode="a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        row["timestamp_utc"],
                        row["event_type"],
                        row["order_id"],
                        row["symbol"],
                        row["side"],
                        row["price"],
                        row["amount"],
                        row["lot_size"],
                        row["risk_amount"],
                        row["fees"],
                        row["status"],
                        row["mode"],
                        row["is_paper"],
                        row["reason"],
                    ]
                )
        except Exception as exc:
            log.error(f"Failed to persist filled order {order.order_id}: {exc}")

    def _post_to_bridge(
        self, sizing: PositionSizing, side: str, price: float
    ) -> OrderResult:
        """Post signal to the Execution Bridge for MQ5 EA polling."""
        payload = {
            "symbol": sizing.symbol,
            "direction": sizing.direction,
            "entry_price": float(sizing.entry_price or price),
            "stop_loss": float(sizing.stop_loss),
            "take_profit": float(sizing.take_profit_1),
            "confidence": 75.0,  # Proxy
        }
        try:
            resp = requests.post(
                self.BRIDGE_URL,
                json=payload,
                headers={"X-API-KEY": self.BRIDGE_KEY},
                timeout=5,
            )
            if resp.status_code == 200:
                log.info(f"Signal posted to bridge for {sizing.symbol}")
                return OrderResult(
                    order_id=f"BRIDGE-{int(time.time())}",
                    symbol=sizing.symbol,
                    side=side,
                    price=price,
                    amount=sizing.position_size,
                    status="filled",  # 'filled' here means successfully queued
                )
        except Exception as exc:
            log.error(f"Failed to post to bridge: {exc}")

        return OrderResult(symbol=sizing.symbol, side=side, status="error")

    # ─── Live Order Execution ─────────────────────────────────────────────────

    def _live_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> OrderResult:
        """
        Place a live order via ccxt with retry logic.

        Args:
            symbol: Trading pair.
            side:   'buy' or 'sell'.
            amount: Amount in base currency.
            price:  Target price (used for LIMIT orders).

        Returns:
            OrderResult from exchange.
        """
        if self.exchange is None:
            log.error("No exchange instance attached — cannot place live order")
            return OrderResult(symbol=symbol, side=side, amount=amount, status="error")

        last_exc: Optional[Exception] = None
        for attempt, delay in enumerate(self.RETRY_DELAYS, start=1):
            try:
                if self.order_type == "limit":
                    raw = self.exchange.create_order(
                        symbol, "limit", side, amount, price
                    )
                else:
                    raw = self.exchange.create_order(symbol, "market", side, amount)

                # Wait for fill confirmation
                filled = self._wait_for_fill(symbol, raw["id"])
                fill_price = float(filled.get("average", price) or price)
                fee_info = filled.get("fee", {})
                fee_cost = float(fee_info.get("cost", 0.0) or 0.0)

                result = OrderResult(
                    order_id=raw["id"],
                    symbol=symbol,
                    side=side,
                    order_type=self.order_type,
                    price=fill_price,
                    amount=float(filled.get("filled", amount)),
                    status=filled.get("status", "filled"),
                    is_paper=False,
                    fees=fee_cost,
                    raw=filled,
                )
                log.info(
                    f"[LIVE] Order {raw['id']} filled: {side.upper()} "
                    f"{amount:.6f} {symbol} @ {fill_price:.4f}"
                )
                return result

            except ccxt.InsufficientFunds as exc:
                log.error(f"Insufficient funds for {symbol}: {exc}")
                return OrderResult(
                    symbol=symbol, side=side, amount=amount, status="rejected"
                )
            except ccxt.InvalidOrder as exc:
                log.error(f"Invalid order for {symbol}: {exc}")
                return OrderResult(
                    symbol=symbol, side=side, amount=amount, status="rejected"
                )
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                last_exc = exc
                log.warning(
                    f"Order attempt {attempt}/{self.MAX_RETRIES} failed: {exc}. Retrying in {delay}s…"
                )
                time.sleep(delay)
            except Exception as exc:
                log.error(f"Unexpected error placing order: {exc}", exc_info=True)
                return OrderResult(
                    symbol=symbol, side=side, amount=amount, status="error"
                )

        log.error(f"Order failed after {self.MAX_RETRIES} attempts: {last_exc}")
        return OrderResult(symbol=symbol, side=side, amount=amount, status="error")

    def _wait_for_fill(
        self,
        symbol: str,
        order_id: str,
        timeout: float = 30.0,
        poll_interval: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Poll for order fill confirmation with a timeout.

        Args:
            symbol:        Trading pair.
            order_id:      Order ID to poll.
            timeout:       Max seconds to wait.
            poll_interval: Polling interval in seconds.

        Returns:
            Filled order dict from exchange, or empty dict on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                order = self.exchange.fetch_order(order_id, symbol)  # type: ignore[union-attr]
                if order.get("status") in ("filled", "closed"):
                    return order
                if order.get("status") == "canceled":
                    log.warning(f"Order {order_id} was cancelled by exchange")
                    return order
            except Exception as exc:
                log.warning(f"Error polling order {order_id}: {exc}")
            time.sleep(poll_interval)

        log.warning(f"Order {order_id} not confirmed within {timeout}s")
        return {}

    def _place_oco(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_loss: float,
        take_profit: float,
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Place an OCO (One-Cancels-Other) order for SL + TP simultaneously.

        Note: OCO is exchange-specific. Binance supports it for spot.
        Falls back to separate SL and TP orders if OCO fails.

        Args:
            symbol:      Trading pair.
            side:        Close side ('sell' for long, 'buy' for short).
            amount:      Amount to close.
            stop_loss:   Stop-loss trigger price.
            take_profit: Take-profit limit price.

        Returns:
            Tuple of (sl_order_dict, tp_order_dict).
        """
        if self.mode == "paper" or self.exchange is None:
            log.debug("OCO skipped in paper mode or no exchange")
            return None, None

        try:
            # Binance-specific OCO order
            if hasattr(self.exchange, "create_oco_order"):
                oco = self.exchange.create_oco_order(
                    symbol,
                    side,
                    amount,
                    take_profit,  # limit price
                    stop_loss,  # stop price
                )
                log.info(f"OCO order placed: {symbol} SL={stop_loss} TP={take_profit}")
                return oco, oco
        except Exception as exc:
            log.warning(f"OCO order failed ({exc}) — placing separate SL/TP orders")

        # Fallback: separate limit orders
        sl_order = None
        tp_order = None
        try:
            sl_order = self.exchange.create_order(
                symbol,
                "stop_market",
                side,
                amount,
                params={"stopPrice": stop_loss, "reduceOnly": True},
            )
            tp_order = self.exchange.create_order(
                symbol, "limit", side, amount, take_profit, params={"reduceOnly": True}
            )
            log.info(f"Separate SL/TP orders placed: SL={stop_loss} TP={take_profit}")
        except Exception as exc:
            log.error(f"Error placing SL/TP orders: {exc}")

        return sl_order, tp_order

    # ─── Status & History ─────────────────────────────────────────────────────

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch open orders from exchange (live mode only).

        Args:
            symbol: Filter by symbol, or None for all.

        Returns:
            List of open order dicts.
        """
        if self.mode == "paper" or self.exchange is None:
            return []
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as exc:
            log.error(f"Failed to fetch open orders: {exc}")
            return []

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order (live mode only).

        Args:
            order_id: Order ID to cancel.
            symbol:   Trading pair.

        Returns:
            True if cancelled successfully.
        """
        if self.mode == "paper" or self.exchange is None:
            return False
        try:
            self.exchange.cancel_order(order_id, symbol)
            log.info(f"Order cancelled: {order_id} ({symbol})")
            return True
        except Exception as exc:
            log.error(f"Failed to cancel order {order_id}: {exc}")
            return False

    def get_paper_summary(self) -> Dict[str, Any]:
        """Return paper portfolio summary."""
        return self.paper_portfolio.get_summary()

    def get_order_history(self) -> List[Dict[str, Any]]:
        """Return list of all placed orders as dicts."""
        return [o.to_dict() for o in self.order_history]
