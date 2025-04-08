import json
import math
from statistics import pstdev
from typing import Any
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str):
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]):
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]):
        compressed = {}
        for symbol, od in order_depths.items():
            compressed[symbol] = [od.buy_orders, od.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]):
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation):
        conversion_observations = {}
        for product, obs in observations.conversionObservations.items():
            conversion_observations[product] = [
                obs.bidPrice,
                obs.askPrice,
                obs.transportFees,
                obs.exportTariff,
                obs.importTariff,
                obs.sugarPrice,
                obs.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]):
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def calculate_vwap(self, order_depth: OrderDepth) -> float:
        """
        Calculate VWAP for the order book, combining buy and sell orders.
        """
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        total_value = 0  # Total (price * volume)
        total_volume = 0  # Total volume

        for price, volume in buy_orders.items():
            total_value += price * volume
            total_volume += volume

        for price, volume in sell_orders.items():
            total_value += price * abs(volume)
            total_volume += abs(volume)

        if total_volume == 0:
            best_bid = max(buy_orders.keys()) if buy_orders else 0
            best_ask = min(sell_orders.keys()) if sell_orders else 999999
            mid = (best_bid + best_ask) / 2.0
            if best_bid == 0 or best_ask == 999999 or mid == float('inf'):
                return 1971.11
            return mid

        return total_value / total_volume

    def run(self, state: TradingState):
        # Load or initialize stored_data
        if state.traderData:
            try:
                stored_data = json.loads(state.traderData)
            except:
                stored_data = {}
        else:
            stored_data = {}

        # Rolling list of VWAP prices (replacing mid prices)
        if "ink_vwap_prices" not in stored_data:
            stored_data["ink_vwap_prices"] = []

        # Track recent trades for PnL calculation
        if "recent_trades" not in stored_data:
            stored_data["recent_trades"] = []

        # Parameters
        SQUID_INK_LIMIT = 50
        rolling_window = 30  # Kept at 30 for stability (yielded 2300 seashells)
        warmup_barrier = 10_000
        warmup_trade_size = 3
        normal_trade_size = 5  # Reduced to limit risk
        stdev_factor_normal = 1.5
        stdev_factor_drawdown = 2.0
        threshold_drawdown = -500
        min_volume = 1  # Minimum volume at best bid/ask to trade
        min_total_volume = 10  # Minimum total volume for VWAP reliability

        # Phase of trading
        timestamp = state.timestamp
        in_warmup = timestamp < warmup_barrier

        # Calculate PnL from recent trades for drawdown detection
        recent_pnl = 0
        for trade in state.own_trades.get("SQUID_INK", []):
            if trade.timestamp >= timestamp - 1000:  # Last 10 timestamps
                if trade.buyer == "SUBMISSION":
                    recent_pnl -= trade.price * trade.quantity  # Bought
                else:
                    recent_pnl += trade.price * trade.quantity  # Sold
        stored_data["recent_trades"].append(recent_pnl)

        # Keep only the last 3 PnL entries for drawdown detection
        if len(stored_data["recent_trades"]) > 3:
            stored_data["recent_trades"] = stored_data["recent_trades"][-3:]
        in_drawdown = sum(stored_data["recent_trades"]) < threshold_drawdown

        # Adjust trade size and stdev factor based on phase
        trade_size = warmup_trade_size if in_warmup else normal_trade_size
        if in_drawdown:
            trade_size = max(1, trade_size // 2)  # Reduce trade size during drawdown
        stdev_factor = stdev_factor_drawdown if in_drawdown else stdev_factor_normal

        result = {}
        conversions = 0

        # ========== SQUID_INK Mean Reversion with VWAP ==========

        if "SQUID_INK" in state.order_depths:
            od = state.order_depths["SQUID_INK"]

            # Calculate VWAP (No window_size argument passed here)
            vwap_price = self.calculate_vwap(od)

            # Check total volume for VWAP reliability
            total_volume = sum(od.buy_orders.values()) + sum(abs(v) for v in od.sell_orders.values())
            if total_volume < min_total_volume:
                vwap_price = 1971.11  # Fallback if insufficient volume

            # Store VWAP price
            stored_data["ink_vwap_prices"].append(vwap_price)
            if len(stored_data["ink_vwap_prices"]) > rolling_window:
                stored_data["ink_vwap_prices"].pop(0)

            # Calculate fair value and thresholds
            mean_vwap = sum(stored_data["ink_vwap_prices"]) / len(stored_data["ink_vwap_prices"])
            if len(stored_data["ink_vwap_prices"]) > 1:
                std_vwap = pstdev(stored_data["ink_vwap_prices"])
            else:
                std_vwap = 1.0

            lower_threshold = mean_vwap - stdev_factor * std_vwap
            upper_threshold = mean_vwap + stdev_factor * std_vwap

            current_position = state.position.get("SQUID_INK", 0)
            ink_orders = []

            # Get best bid and ask for trading
            best_bid = max(od.buy_orders.keys()) if od.buy_orders else 0
            best_ask = min(od.sell_orders.keys()) if od.sell_orders else 999999

            # Debug logging
            logger.print(f"SQUID_INK: timestamp={timestamp}, in_warmup={in_warmup}, in_drawdown={in_drawdown}")
            logger.print(f"SQUID_INK: best_bid={best_bid}, best_ask={best_ask}, vwap_price={vwap_price}")
            logger.print(f"SQUID_INK: mean_vwap={mean_vwap}, std_vwap={std_vwap}")
            logger.print(f"SQUID_INK: lower_threshold={lower_threshold}, upper_threshold={upper_threshold}")
            logger.print(f"SQUID_INK: position={current_position}, trade_size={trade_size}")

            # Buy condition
            if (best_ask < lower_threshold and best_ask != 999999 and best_ask in od.sell_orders and
                abs(od.sell_orders[best_ask]) >= min_volume):
                can_buy = SQUID_INK_LIMIT - current_position
                if can_buy > 0:
                    ask_volume = abs(od.sell_orders[best_ask])
                    buy_qty = min(can_buy, ask_volume, trade_size)
                    if buy_qty > 0:
                        ink_orders.append(Order("SQUID_INK", best_ask, buy_qty))
                        logger.print(f"SQUID_INK: Placing BUY order for {buy_qty} at {best_ask}")
                    else:
                        logger.print("SQUID_INK: Buy quantity is 0, no trade")
                else:
                    logger.print("SQUID_INK: Cannot buy, position limit reached")
            else:
                logger.print("SQUID_INK: Buy condition not met")

            # Sell condition
            if (best_bid > upper_threshold and best_bid != 0 and best_bid in od.buy_orders and
                od.buy_orders[best_bid] >= min_volume):
                can_sell = SQUID_INK_LIMIT + current_position
                if can_sell > 0:
                    bid_volume = od.buy_orders[best_bid]
                    sell_qty = min(can_sell, bid_volume, trade_size)
                    if sell_qty > 0:
                        ink_orders.append(Order("SQUID_INK", best_bid, -sell_qty))
                        logger.print(f"SQUID_INK: Placing SELL order for {sell_qty} at {best_bid}")
                    else:
                        logger.print("SQUID_INK: Sell quantity is 0, no trade")
                else:
                    logger.print("SQUID_INK: Cannot sell, position limit reached")
            else:
                logger.print("SQUID_INK: Sell condition not met")

            if ink_orders:
                result["SQUID_INK"] = ink_orders

        trader_data = json.dumps(stored_data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
