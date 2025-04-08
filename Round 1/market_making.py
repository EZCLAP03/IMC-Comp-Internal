import math
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple, Any
import numpy as np

def poisson_trade_amount(mean: float) -> int:

    return np.random.poisson(mean)

# -------------------------------------------------------------------
# Logger: collects log messages and flushes them in the required text format.
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Status: encapsulates market data for KELP.
# Position limit for KELP is 50.
# -------------------------------------------------------------------
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
    def maxamt_midprc(self) -> float:
        buy_orders = self._state.order_depths[self.product].buy_orders
        sell_orders = self._state.order_depths[self.product].sell_orders
        if not buy_orders or not sell_orders:
            return (self.best_bid + self.best_ask) / 2.0
        max_bv = 0
        max_bv_price = self.best_bid
        for p, v in buy_orders.items():
            if v > max_bv:
                max_bv = v
                max_bv_price = p
        max_sv = 0
        max_sv_price = self.best_ask
        for p, v in sell_orders.items():
            if -v > max_sv:
                max_sv = -v
                max_sv_price = p
        return (max_bv_price + max_sv_price) / 2

    @property
    def vwap(self) -> float:
        """
        Calculate Volume Weighted Average Price (VWAP) for the product.
        Combines bid and ask data.
        """
        buy_orders = self._state.order_depths[self.product].buy_orders
        sell_orders = self._state.order_depths[self.product].sell_orders

        total_value = 0  # Total (price * volume)
        total_volume = 0  # Total volume

        # Aggregate bid data
        for price, volume in buy_orders.items():
            total_value += price * volume
            total_volume += volume

        # Aggregate ask data
        for price, volume in sell_orders.items():
            total_value += price * abs(volume)
            total_volume += abs(volume)

        # Prevent division by zero
        if total_volume == 0:
            return (self.best_bid + self.best_ask) / 2.0  # Default to mid-price

        return total_value / total_volume


# -------------------------------------------------------------------
# Trade: identifies the highest-volume bid and ask and undercuts them by 1.
# Order sizes are capped by our remaining capacity.
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Trader: main entry point.
# -------------------------------------------------------------------
class Trader:
    def __init__(self):
        pass

    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int,
                                position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)  # max to buy
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)  # max we can sell
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity

        buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit,
                                                                        "RAINFOREST_RESIN", buy_order_volume,
                                                                        sell_order_volume, fair_value, 1)

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))  # buy

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))  # sell

        return orders

    @staticmethod
    def kelp(state: Status) -> List[Order]:
        orders = []

        vwap_price = state.vwap  # Use VWAP as the fair price

        if state.bids:
            best_bid_level = max(state.bids, key=lambda x: x[1])  # (price, volume)
            bid_price, bid_vol = best_bid_level
            buy_qty = min(state.possible_buy_amt, bid_vol)
            if buy_qty > 0 and bid_price < vwap_price:  # Ensure we buy below VWAP
                orders.append(Order(state.product, bid_price + 1, buy_qty))

        if state.asks:
            best_ask_level = max(state.asks, key=lambda x: abs(x[1]))
            ask_price, ask_vol = best_ask_level
            sell_qty = min(state.possible_sell_amt, abs(ask_vol))
            if sell_qty > 0 and ask_price > vwap_price:  # Ensure we sell above VWAP
                orders.append(Order(state.product, ask_price - 1, -sell_qty))

        return orders

    @staticmethod
    def squid_ink(state: Status) -> List[Order]:
        orders = []
        mid_price = (state.best_bid + state.best_ask) / 2.0

        if state.bids:
            best_bid_level = max(state.bids, key=lambda x: x[1])  # (price, volume)
            bid_price, bid_vol = best_bid_level
            buy_qty = min(state.possible_buy_amt, bid_vol)

            # Ensure the bid is sufficiently away from the mid-price
            if buy_qty > 0 and bid price < mid_price:
                orders.append(Order(state.product, bid_price + 1, buy_qty))

        if state.asks:
            best_ask_level = max(state.asks, key=lambda x: abs(x[1]))
            ask_price, ask_vol = best_ask_level
            sell_qty = min(state.possible_sell_amt, abs(ask_vol))

            # Ensure the ask is sufficiently away from the mid-price
            if sell_qty > 0 and abs(ask_price - mid_price) > 1:
                orders.append(Order(state.product, ask_price - 1, -sell_qty))

        return orders

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int,
                             product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float,
                             width: int) -> Tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # Poisson-based size adjustment for critical positions near limits
        close_to_limits = abs(position) >= 0.90 * position_limit  # 15% margin to the position limits
        if close_to_limits:
            mean_trade_size = 5  # Arbitrary small trade size mean for Poisson
            if position > 0:  # If we are net long
                offload_quantity = min(poisson_trade_amount(mean_trade_size), sell_quantity)
                orders.append(Order(product, fair_for_ask, -offload_quantity))
                sell_order_volume += offload_quantity
            elif position < 0:  # If we are net short
                offload_quantity = min(poisson_trade_amount(mean_trade_size), buy_quantity)
                orders.append(Order(product, fair_for_bid, offload_quantity))
                buy_order_volume += offload_quantity

        # Zero EV trades to clear position
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
        rainforest_resin_position_limit = 50

        if "RAINFOREST_RESIN" in state.order_depths:
            order_depth = state.order_depths["RAINFOREST_RESIN"]

            if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                rainforest_resin_width = (best_ask - best_bid) / 2  # Half of the spread
            else:
                rainforest_resin_width = 5  # Default fallback value

            total_value = 0
            total_volume = 0

            for price, volume in order_depth.buy_orders.items():
                total_value += price * volume
                total_volume += volume

            for price, volume in order_depth.sell_orders.items():
                total_value += price * abs(volume)  # absolute volume for sellers
                total_volume += abs(volume)

            if total_volume > 0:
                rainforest_resin_fair_value = total_value / total_volume
            else:
                rainforest_resin_fair_value = 10000  # fallback

            rainforest_resin_position = state.position.get("RAINFOREST_RESIN", 0)

            # Apply updated Poisson-based and 0EV position management here
            rainforest_resin_orders = self.rainforest_resin_orders(
                order_depth,
                rainforest_resin_fair_value,
                rainforest_resin_width,
                rainforest_resin_position,
                rainforest_resin_position_limit
            )

            result["RAINFOREST_RESIN"] = rainforest_resin_orders

        if "KELP" in state.order_depths:
            kelp_status = Status("KELP", state)
            kelp_orders = Trader.kelp(kelp_status)
            result["KELP"] = kelp_orders

        if "SQUID_INK" in state.order_depths:
            squid_ink_status = Status("SQUID_INK", state)
            squid_ink_orders = Trader.squid_ink(squid_ink_status)
            result["SQUID_INK"] = squid_ink_orders

        traderData = "Processed both RAINFOREST_RESIN and KELP orders."
        conversions = 1

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
