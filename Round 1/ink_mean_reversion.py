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
    def run(self, state: TradingState):
        # Load or initialize stored_data
        if state.traderData:
            try:
                stored_data = json.loads(state.traderData)
            except:
                stored_data = {}
        else:
            stored_data = {}

        # Rolling list of "mid prices"
        if "ink_mid_prices" not in stored_data:
            stored_data["ink_mid_prices"] = []

        # Track recent PnL or recent trades to detect drawdown
        # (We only have partial info in this example, but let's assume we do if needed.)
        if "recent_pnl" not in stored_data:
            stored_data["recent_pnl"] = []

        # We'll assume the "pnl" might come from analyzing own_trades or something else
        # For the sake of example, we'll skip real PnL extraction.

        # We do store the last known "net trade result" if you have a method to track that
        # (not detailed in base code). You could store realized PnL from own_trades.

        # Basic parameters
        SQUID_INK_LIMIT = 50
        rolling_window = 30  # for mid prices
        fallback_mid = 1000.0

        # Additional "warm-up" approach
        warmup_barrier = (
            10_000  # e.g. skip first 10k timestamps or trade with smaller size
        )
        warmup_trade_size = 3
        normal_trade_size = 10

        # If big drop in PnL or big losing streak, we might up the stdev factor or reduce trade size
        stdev_factor_normal = 1.5
        stdev_factor_drawdown = 2.0  # if we detect big drop
        threshold_drawdown = (
            -500
        )  # e.g. if we lose 500 in short timeframe => bigger stdev factor

        # We'll do partial logic to see if we are in warmup or in meltdown
        timestamp = state.timestamp

        # 1) Figure out "phase" of trading
        in_warmup = timestamp < warmup_barrier
        # We'll assume "in_drawdown" if last 3 trades net is < threshold_drawdown => user can refine
        in_drawdown = False  # For demonstration only

        # Decide trade size
        trade_size = warmup_trade_size if in_warmup else normal_trade_size

        # Decide stdev factor
        stdev_factor = stdev_factor_drawdown if in_drawdown else stdev_factor_normal

        result = {}
        conversions = 0

        # ========== Chỉ giao dịch SQUID_INK, theo mean reversion ==========

        if "SQUID_INK" in state.order_depths:
            od = state.order_depths["SQUID_INK"]

            # Tính mid price
            best_bid = 0
            if od.buy_orders:
                best_bid = max(od.buy_orders.keys())

            best_ask = 999999
            if od.sell_orders:
                best_ask = min(od.sell_orders.keys())

            mid_price = fallback_mid
            if best_bid > 0 and best_ask < 999999:
                mid_price = (best_bid + best_ask) / 2.0

            # Lưu mid vào chuỗi
            stored_data["ink_mid_prices"].append(mid_price)
            if len(stored_data["ink_mid_prices"]) > rolling_window:
                stored_data["ink_mid_prices"].pop(0)

            mean_mid = sum(stored_data["ink_mid_prices"]) / len(
                stored_data["ink_mid_prices"]
            )
            if len(stored_data["ink_mid_prices"]) > 1:
                std_mid = pstdev(stored_data["ink_mid_prices"])
            else:
                std_mid = 1.0

            lower_threshold = mean_mid - stdev_factor * std_mid
            upper_threshold = mean_mid + stdev_factor * std_mid

            current_position = state.position.get("SQUID_INK", 0)
            ink_orders = []

            # -- If in warm-up, we can skip trades or do smaller trades
            # Already handled by smaller trade_size. Also can skip entirely if you prefer:
            # if in_warmup:
            #     # do nothing => pass => won't trade
            # else:
            #     # do actual trading

            # ========== Condition BUY ==========
            if best_ask < lower_threshold and best_ask != 999999:
                can_buy = SQUID_INK_LIMIT - current_position
                if can_buy > 0 and best_ask in od.sell_orders:
                    ask_volume = abs(od.sell_orders[best_ask])
                    buy_qty = min(can_buy, ask_volume, trade_size)
                    if buy_qty > 0:
                        ink_orders.append(Order("SQUID_INK", best_ask, buy_qty))

            # ========== Condition SELL ==========
            if best_bid > upper_threshold and best_bid != 0:
                can_sell = SQUID_INK_LIMIT + current_position
                if can_sell > 0 and best_bid in od.buy_orders:
                    bid_volume = od.buy_orders[best_bid]
                    sell_qty = min(can_sell, bid_volume, trade_size)
                    if sell_qty > 0:
                        ink_orders.append(Order("SQUID_INK", best_bid, -sell_qty))

            if ink_orders:
                result["SQUID_INK"] = ink_orders

        trader_data = json.dumps(stored_data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
