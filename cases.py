"""课程样例和固定种子的自设场景。"""

import random

from game_search import TreeNode


MAZE = [list("S..X."), list(".X..."), list(".X.X."), list("...XE"), list("XX...")]
WEIGHTED = [["S", 1, 1, 8, 1], [1, 8, 3, 1, 1], [1, 8, 1, 8, 1],
            [1, 1, 1, 8, 1], [8, 8, 1, 1, "E"]]
TREASURE_MAP = [["A", 1, 1, "X", 2, 1, "T1"], [1, "X", 6, "X", 1, "X", 1],
                [1, 1, "T2", 1, 1, 1, 1], ["X", 1, "X", 8, "X", 1, "X"],
                [1, 1, 1, 1, "T3", 1, 1], [1, "X", 9, "X", 1, "X", 1],
                ["T4", 1, 1, "X", 1, 1, "B"]]
TREASURE_VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}


def example_tree():
    left = TreeNode(children=[TreeNode(3, name="L00"), TreeNode(5, name="L01")], name="M0")
    shared = TreeNode(children=[TreeNode(6, name="L10"), TreeNode(9, name="L11")], name="M1")
    right = TreeNode(children=[TreeNode(1, name="L20"), TreeNode(2, name="L21")], name="M2")
    # 按题目构造代码复用中间节点；搜索按两次到达分别计数。
    return TreeNode(children=[TreeNode(children=[left, shared], name="N0"),
                              TreeNode(children=[shared, right], name="N1")], name="root")


def random_tree(branching, depth, seed):
    rng = random.Random(seed)

    def build(level, name):
        if level == depth:
            return TreeNode(rng.randint(-20, 20), name=name)
        return TreeNode(children=[build(level + 1, name + "/" + str(i)) for i in range(branching)], name=name)

    return build(0, "r")


def path_scenarios():
    scenarios = []
    for size in (20, 40, 80):
        scenarios.append(dict(name=f"open_{size}", family="open", size=size, seed=None,
                              grid=[[1] * size for _ in range(size)], weighted=False))
        for seed in (66, 88):
            for family, density, weighted in (("obstacle", 0.25, False), ("terrain", 0.15, True)):
                rng = random.Random(seed + size * 100)
                grid = [["X" if rng.random() < density else
                         rng.choice([1, 1, 1, 3, 8]) if weighted else 1
                         for _ in range(size)] for _ in range(size)]
                grid[0][0] = grid[-1][-1] = 1
                scenarios.append(dict(name=f"{family}_{size}_{seed}", family=family,
                                      size=size, seed=seed, grid=grid, weighted=weighted))
    grid = [[1] * 20 for _ in range(20)]
    grid[10] = ["X"] * 20
    scenarios.append(dict(name="blocked_20", family="blocked", size=20, seed=None, grid=grid, weighted=False))
    return scenarios


def treasure_candidate(seed, index):
    rng = random.Random(seed * 1000 + index)
    grid = [["X" if rng.random() < 0.12 else rng.choice([1, 1, 1, 2, 5])
             for _ in range(7)] for _ in range(7)]
    positions = {"A": (0, 0), "B": (6, 6), "T1": (0, 6), "T2": (2, 2),
                 "T3": (4, 4), "T4": (6, 0)}
    for name, (r, c) in positions.items():
        grid[r][c] = name
    values = {name: rng.randint(6, 14) for name in positions if name.startswith("T")}
    return grid, values
