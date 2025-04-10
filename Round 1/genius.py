import math
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple, Any
import numpy as np
import statistics
import json

def poisson_trade_amount(mean: float) -> int:
    return np.random.poisson(mean)

# Logger class from the second code
class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[str, List[Order]], conversions: int, trader_data: str) -> None:
        lines = []
        lines.append(f"timestamp: {state.timestamp}")
        for product, position in state.position.items():
            lines.append(f"Position for {product}: {position}")
        lines.append(trader_data)
        for symbol, order_list in orders.items():
            for o in order_list:
                if o.quantity > 0:
                    lines.append(f"BUY Order: {o.quantity} x {symbol} at {o.price}")
                elif o.quantity < 0:
                    lines.append(f"SELL Order: {abs(o.quantity)} x {symbol} at {o.price}")
        self.logs += "\n".join(lines) + "\n"
        print(self.logs)
        self.logs = ""

logger = Logger()

# Status class from the second code
class Status:
    def __init__(self, product: str, state: TradingState) -> None:
        self.product = product
        self._state = state

    @property
    def order_depth(self) -> OrderDepth:
        return self._state.order_depths[self.product]

    @property
    def bids(self) -> List[Tuple[int, int]]:
        return list(self._state.order_depths[self.product].buy_orders.items())

    @property
    def asks(self) -> List[Tuple[int, int]]:
        return list(self._state.order_depths[self.product].sell_orders.items())

    @property
    def position(self) -> int:
        return self._state.position.get(self.product, 0)

    @property
    def possible_buy_amt(self) -> int:
        return 50 - self.position

    @property
    def possible_sell_amt(self) -> int:
        return 50 + self.position

    @property
    def best_bid(self) -> int:
        bids = self._state.order_depths[self.product].buy_orders
        return max(bids.keys()) if bids else 0

    @property
    def best_ask(self) -> int:
        asks = self._state.order_depths[self.product].sell_orders
        return min(asks.keys()) if asks else float('inf')

    @property
    def vwap(self) -> float:
        buy_orders = self._state.order_depths[self.product].buy_orders
        sell_orders = self._state.order_depths[self.product].sell_orders
        total_value = 0
        total_volume = 0
        for price, volume in buy_orders.items():
            total_value += price * volume
            total_volume += volume
        for price, volume in sell_orders.items():
            total_value += price * abs(volume)
            total_volume += abs(volume)
        if total_volume == 0:
            return (self.best_bid + self.best_ask) / 2.0
        return total_value / total_volume

# Trader class with integrated strategies
class Trader:
    def __init__(self):
        # Parameters for SQUID_INK from the first code
        self.position_limit = 50
        self.window = 20
        self.alpha = 0.35
        self.vol_factor = 1.2
        self.min_tick = 1.0
        self.max_hold_duration = 10

    def squid_ink(self, state: TradingState) -> List[Order]:
        # Exact SQUID_INK logic from the first code's run method
        data = json.loads(state.traderData) if state.traderData else {}
        mid_prices = data.get("mid_prices", [])
        ema_price = data.get("ema_price", None)
        hold_counter = data.get("hold_counter", 0)

        od: OrderDepth = state.order_depths.get("SQUID_INK", OrderDepth())
        if not od.buy_orders or not od.sell_orders:
            return []

        best_bid, best_ask = max(od.buy_orders), min(od.sell_orders)
        bid_vol, ask_vol = od.buy_orders[best_bid], -od.sell_orders[best_ask]
        mid_price = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

        mid_prices.append(mid_price)
        if len(mid_prices) > self.window:
            mid_prices.pop(0)

        if ema_price is None:
            ema_price = mid_price
        else:
            ema_price = self.alpha * mid_price + (1 - self.alpha) * ema_price

        std_dev = statistics.pstdev(mid_prices) if len(mid_prices) > 1 else 1.0

        position = state.position.get("SQUID_INK", 0)
        orders = []
        arb_threshold = 0.5 * std_dev

        # Adaptive Spread
        spread = max(0.2 * std_dev, self.min_tick * 2)
        half_spread = spread / 2
        bid_target = math.floor(ema_price - half_spread)
        ask_target = math.ceil(ema_price + half_spread)

        # Arbitrage logic
        for ask, vol in sorted(od.sell_orders.items()):
            if ask < ema_price - arb_threshold and position < self.position_limit:
                qty = min(-vol, self.position_limit - position)
                orders.append(Order("SQUID_INK", ask, qty))
                position += qty

        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            if bid > ema_price + arb_threshold and position > -self.position_limit:
                qty = min(vol, position + self.position_limit)
                orders.append(Order("SQUID_INK", bid, -qty))
                position -= qty

        # Stop-loss logic
        if abs(position) >= 0.8 * self.position_limit:
            hold_counter += 1
        else:
            hold_counter = 0

        if hold_counter >= self.max_hold_duration:
            reduce_qty = int(abs(position) * 0.5)
            if position > 0:
                orders.append(Order("SQUID_INK", math.floor(ema_price), -reduce_qty))
                position -= reduce_qty
            elif position < 0:
                orders.append(Order("SQUID_INK", math.ceil(ema_price), reduce_qty))
                position += reduce_qty
            hold_counter = 0

        # Market Making Adaptive
        available_buy = self.position_limit - position
        available_sell = self.position_limit + position

        buy_qty = min(available_buy, 10) if available_buy > 0 else 0
        sell_qty = min(available_sell, 10) if available_sell > 0 else 0

        if buy_qty > 0 and bid_target < best_ask:
            orders.append(Order("SQUID_INK", bid_target, buy_qty))

        if sell_qty > 0 and ask_target > best_bid:
            orders.append(Order("SQUID_INK", ask_target, -sell_qty))

        # Update trader_data
        data.update({
            "mid_prices": mid_prices,
            "ema_price": ema_price,
            "hold_counter": hold_counter,
        })
        state.traderData = json.dumps(data)

        logger.print(
            f"Mid:{mid_price:.2f},EMA:{ema_price:.2f},Std:{std_dev:.2f},Pos:{position},Hold:{hold_counter}"
        )

        return orders

    @staticmethod
    def kelp(state: Status) -> List[Order]:
        # Unchanged KELP strategy from the second code
        orders = []
        vwap_price = state.vwap
        if state.bids:
            best_bid_level = max(state.bids, key=lambda x: x[1])
            bid_price, bid_vol = best_bid_level
            buy_qty = min(state.possible_buy_amt, bid_vol)
            if buy_qty > 0 and bid_price < vwap_price:
                orders.append(Order(state.product, bid_price + 1, buy_qty))
        if state.asks:
            best_ask_level = max(state.asks, key=lambda x: abs(x[1]))
            ask_price, ask_vol = best_ask_level
            sell_qty = min(state.possible_sell_amt, abs(ask_vol))
            if sell_qty > 0 and ask_price > vwap_price:
                orders.append(Order(state.product, ask_price - 1, -sell_qty))
        return orders

    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int,
                                position_limit: int) -> List[Order]:
        # Unchanged RAINFOREST_RESIN strategy from the second code
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_order_volume += quantity
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity
        buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit,
                                                                        "RAINFOREST_RESIN", buy_order_volume,
                                                                        sell_order_volume, fair_value, 1)
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))
        return orders

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int,
                             product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float,
                             width: int) -> Tuple[int, int]:
        # Unchanged from the second code
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)
        close_to_limits = abs(position) >= 0.90 * position_limit
        if close_to_limits:
            mean_trade_size = max(1, abs(position) // 10)
            if position > 0:
                offload_quantity = min(poisson_trade_amount(mean_trade_size), sell_quantity)
                orders.append(Order(product, fair_for_ask, -offload_quantity))
                sell_order_volume += offload_quantity
            elif position < 0:
                offload_quantity = min(poisson_trade_amount(mean_trade_size), buy_quantity)
                orders.append(Order(product, fair_for_bid, offload_quantity))
                buy_order_volume += offload_quantity
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # SQUID_INK strategy
        if "SQUID_INK" in state.order_depths:
            squid_ink_orders = self.squid_ink(state)
            result["SQUID_INK"] = squid_ink_orders
            traderData = state.traderData  # Updated within squid_ink
        else:
            traderData = state.traderData if state.traderData else ""

        # KELP strategy
        if "KELP" in state.order_depths:
            kelp_status = Status("KELP", state)
            kelp_orders = self.kelp(kelp_status)
            result["KELP"] = kelp_orders

        # RAINFOREST_RESIN strategy
        if "RAINFOREST_RESIN" in state.order_depths:
            order_depth = state.order_depths["RAINFOREST_RESIN"]
            position = state.position.get("RAINFOREST_RESIN", 0)
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
            fair_value = (best_bid + best_ask) / 2 if best_bid and best_ask < float('inf') else 0
            width = 1
            position_limit = 50
            resin_orders = self.rainforest_resin_orders(order_depth, fair_value, width, position, position_limit)
            result["RAINFOREST_RESIN"] = resin_orders

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData