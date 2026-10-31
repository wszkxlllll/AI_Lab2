"""宝藏地图、玩家状态和有限深度博弈。"""

from dataclasses import dataclass
import math

from astar import astar_grid
from game_search import search_game


@dataclass(frozen=True)
class State:
    pos_a: str
    pos_b: str
    remaining: tuple
    score_a: float = 0
    score_b: float = 0
    player: str = "A"

    @property
    def utility(self):
        return self.score_a - self.score_b


def locate(grid):
    positions = {}
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value in ("A", "B") or str(value).startswith("T"):
                if value in positions:
                    raise ValueError("玩家和宝藏标记不能重复")
                positions[value] = (r, c)
    return positions


def distance_matrix(grid):
    """每个有向位置对独立运行 A*，不提前假设距离对称。"""
    positions = locate(grid)
    names = ["A", "B"] + sorted(n for n in positions if n.startswith("T"))
    if "A" not in positions or "B" not in positions:
        raise ValueError("地图需要标出 A 和 B")
    distances, details = {}, []
    for source in names:
        distances[source] = {}
        for target in names:
            if source == target:
                distances[source][target] = 0
                continue
            result = astar_grid(positions[source], positions[target], grid, weighted=True)
            distances[source][target] = result.cost
            details.append(dict(source=source, target=target, cost=result.cost,
                                expanded=result.expanded, path=result.path))
    return names, distances, details


class TreasureGame:
    def __init__(self, distances, values):
        self.distances = distances
        self.values = dict(values)
        self.initial = State("A", "B", tuple(sorted(values)))
        names = ["A", "B"] + list(values)
        if any(not math.isfinite(distances[a][b]) for a in names for b in names):
            raise ValueError("本实验的宝藏场景要求所有标记位置互相可达")

    def gain(self, state, target, player=None):
        player = state.player if player is None else player
        position = state.pos_a if player == "A" else state.pos_b
        return self.values[target] - self.distances[position][target]

    def step(self, state, target):
        if target not in state.remaining:
            raise ValueError("只能选择尚未取得的宝藏")
        rest = tuple(t for t in state.remaining if t != target)
        gain = self.gain(state, target)
        if state.player == "A":
            return State(target, state.pos_b, rest, state.score_a + gain, state.score_b, "B")
        return State(state.pos_a, target, rest, state.score_a, state.score_b + gain, "A")

    def evaluate(self, state, mode="current"):
        if mode == "current" or not state.remaining:
            return state.utility
        if mode != "opportunity":
            raise ValueError("评价方式应为 current 或 opportunity")
        # 当前最佳剩余收益差作为未来机会估计，系数固定为 1。
        opportunity_a = max(self.gain(state, t, "A") for t in state.remaining)
        opportunity_b = max(self.gain(state, t, "B") for t in state.remaining)
        return state.utility + opportunity_a - opportunity_b

    def ordering(self, state, moves, mode):
        if mode == "natural":
            return moves
        if mode == "reverse":
            return moves[::-1]
        if mode == "gain":
            return sorted(moves, key=lambda t: (-self.gain(state, t), t))
        raise ValueError("宝藏顺序应为 natural、reverse 或 gain")

    def solve(self, depth, pruning=True, order="natural", evaluation="current", state=None, trace=False):
        initial = self.initial if state is None else state
        return search_game(initial, lambda s: s.remaining, self.step,
                           lambda s: self.evaluate(s, evaluation), lambda s: s.player == "A",
                           depth=depth, pruning=pruning,
                           ordering=lambda s, a: self.ordering(s, a, order), trace=trace)

    def complete_after(self, target):
        """固定首步后，双方按完整 Minimax 选择剩余行动。"""
        state = self.step(self.initial, target)
        return self.solve(len(state.remaining), state=state).value

    def trajectory(self, moves):
        state, rows = self.initial, []
        for turn, target in enumerate(moves, 1):
            player = state.player
            source = state.pos_a if player == "A" else state.pos_b
            gain = self.gain(state, target)
            state = self.step(state, target)
            rows.append(dict(turn=turn, player=player, source=source, target=target,
                             distance=self.distances[source][target], gain=gain,
                             score_a=state.score_a, score_b=state.score_b, utility=state.utility))
        return rows


def solve_treasure(grid, values, depth, pruning=True):
    names, distances, _ = distance_matrix(grid)
    result = TreasureGame(distances, values).solve(depth, pruning=pruning)
    value = int(result.value) if float(result.value).is_integer() else result.value
    return {"best_target": result.best_action, "best_value": value}


if __name__ == "__main__":
    import json
    import sys
    data = json.load(sys.stdin)
    print(json.dumps(solve_treasure(data["grid"], data["values"], data["H"])))
