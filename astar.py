"""A* 寻路、启发函数和搜索过程统计。"""

from collections import deque
from dataclasses import dataclass, field
import heapq
import itertools
import math


@dataclass
class PathResult:
    path: list
    cost: float
    expanded: int = 0
    pushed: int = 0
    popped: int = 0
    stale: int = 0
    reopened: int = 0
    peak_open: int = 0
    trace: list = field(default_factory=list)


def astar_graph(start, goal, neighbors, heuristic, tie="larger_g", reopen=True, trace=False):
    """在非负边权图上搜索；发现更短路径时更新前驱。"""
    if tie not in {"fifo", "larger_g", "smaller_g"}:
        raise ValueError("同分规则应为 fifo、larger_g 或 smaller_g")
    result = PathResult([], math.inf)
    distance, parent, closed = {start: 0}, {}, set()
    queue, sequence = [], itertools.count()

    def push(node, cost):
        secondary = -cost if tie == "larger_g" else cost if tie == "smaller_g" else 0
        heapq.heappush(queue, (cost + heuristic(node), secondary, next(sequence), cost, node))
        result.pushed += 1
        result.peak_open = max(result.peak_open, len(queue))

    push(start, 0)
    while queue:
        f, _, _, g, node = heapq.heappop(queue)
        if g != distance.get(node) or node in closed:
            result.stale += 1
            continue
        result.popped += 1
        if trace:
            result.trace.append(dict(event="取出", node=node, g=g, h=heuristic(node), f=f))
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            result.path, result.cost = path[::-1], g
            return result
        closed.add(node)
        result.expanded += 1
        for neighbor, weight in neighbors(node):
            if weight < 0 or not math.isfinite(weight):
                raise ValueError("A* 要求边权为有限非负数")
            candidate = g + weight
            if candidate >= distance.get(neighbor, math.inf):
                continue
            if neighbor in closed:
                if not reopen:
                    continue
                closed.remove(neighbor)
                result.reopened += 1
                if trace:
                    result.trace.append(dict(event="重开", node=neighbor,
                                             old_g=distance[neighbor], new_g=candidate))
            distance[neighbor], parent[neighbor] = candidate, node
            push(neighbor, candidate)
    return result


def terrain_costs(grid, weighted=False, goal=None, goal_free=False):
    """把字符地图转成进入代价；起点初始代价另设为零。"""
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("地图必须是非空矩形")
    costs = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "X":
                continue
            if not weighted or cell in ("S", "E", "A", "B", ".") or str(cell).startswith("T"):
                value = 1
            else:
                value = float(cell)
            if value < 0 or not math.isfinite(value):
                raise ValueError("地形代价必须是有限非负数")
            costs[(r, c)] = value
    if goal_free and goal in costs:
        costs[goal] = 0
    return costs


def adjacent(node, costs):
    r, c = node
    for target in ((r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)):
        if target in costs:
            yield target, costs[target]


def manhattan(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def make_heuristic(costs, goal, mode="safe"):
    if mode == "zero":
        return lambda node: 0
    if mode == "manhattan":
        return lambda node: manhattan(node, goal)
    if mode != "safe":
        raise ValueError("启发函数应为 zero、manhattan 或 safe")
    lower = min((v for p, v in costs.items() if p != goal), default=0)
    # 最后一步单独计价，因而能同时处理免费终点和普通终点。
    return lambda node: (0 if node == goal else
                         lower * max(manhattan(node, goal) - 1, 0) + costs[goal])


def astar_grid(start, goal, grid, weighted=False, goal_free=False,
               heuristic="safe", tie="larger_g", trace=False):
    start, goal = tuple(start), tuple(goal)
    costs = terrain_costs(grid, weighted, goal, goal_free)
    if start not in costs or goal not in costs:
        return PathResult([], math.inf)
    return astar_graph(start, goal, lambda n: adjacent(n, costs),
                       make_heuristic(costs, goal, heuristic), tie=tie, trace=trace)


def bfs_grid(start, goal, grid):
    """单位代价地图的广度优先参照。"""
    start, goal = tuple(start), tuple(goal)
    costs = terrain_costs(grid)
    if start not in costs or goal not in costs:
        return PathResult([], math.inf)
    queue, parent, seen = deque([start]), {}, {start}
    result = PathResult([], math.inf, pushed=1, peak_open=1)
    while queue:
        node = queue.popleft()
        result.popped += 1
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            result.path, result.cost = path[::-1], len(path) - 1
            return result
        result.expanded += 1
        for nxt, _ in adjacent(node, costs):
            if nxt not in seen:
                seen.add(nxt)
                parent[nxt] = node
                queue.append(nxt)
                result.pushed += 1
                result.peak_open = max(result.peak_open, len(queue))
    return result


def distances_to_goal(costs, goal):
    """反向 Dijkstra：反向边仍支付原正向边的进入代价。"""
    distance, queue = {goal: 0}, [(0, goal)]
    while queue:
        value, node = heapq.heappop(queue)
        if value != distance[node]:
            continue
        for previous, _ in adjacent(node, costs):
            candidate = value + costs[node]
            if candidate < distance.get(previous, math.inf):
                distance[previous] = candidate
                heapq.heappush(queue, (candidate, previous))
    return distance


def solve_maze(start, end, grid):
    return astar_grid(start, end, grid).path


def solve_weighted(start, end, grid):
    cost = astar_grid(start, end, grid, weighted=True, goal_free=True).cost
    return -1 if math.isinf(cost) else int(cost) if float(cost).is_integer() else cost


if __name__ == "__main__":
    import argparse
    import json
    import sys
    parser = argparse.ArgumentParser(description="读取标准输入中的 JSON 地图")
    parser.add_argument("task", choices=["maze", "weighted"])
    args = parser.parse_args()
    data = json.load(sys.stdin)
    solve = solve_maze if args.task == "maze" else solve_weighted
    print(json.dumps(solve(data["start"], data["end"], data["grid"])))
