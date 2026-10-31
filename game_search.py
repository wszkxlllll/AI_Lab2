"""Minimax、Alpha-beta 与博弈树访问记录。"""

from dataclasses import dataclass, field
import math
import random
import zlib


@dataclass
class GameResult:
    value: float
    best_action: object = None
    principal_variation: list = field(default_factory=list)
    visited: int = 0
    leaves: int = 0
    cutoffs: int = 0
    skipped_children: int = 0
    events: list = field(default_factory=list)


def search_game(initial, actions, transition, evaluate, is_max, depth=None,
                pruning=True, ordering=None, trace=False):
    """相同递归分别运行完整搜索和剪枝搜索，计数包含根节点。"""
    if depth is not None and depth < 0:
        raise ValueError("搜索深度不能为负数")
    result = GameResult(0)

    def visit(state, remaining, alpha, beta, path):
        result.visited += 1
        alpha_in, beta_in = alpha, beta
        moves = list(actions(state))
        if remaining == 0 or not moves:
            value = evaluate(state)
            result.leaves += 1
            if trace:
                result.events.append(dict(event="叶子", path=path, value=value,
                                          alpha=alpha, beta=beta))
            return value, None, []
        maximizing = is_max(state)
        if ordering is not None:
            moves = ordering(state, moves)
        best = -math.inf if maximizing else math.inf
        chosen, variation = None, []
        for i, action in enumerate(moves):
            child = transition(state, action)
            value, _, continuation = visit(child, None if remaining is None else remaining - 1,
                                            alpha, beta, path + "/" + str(action))
            if chosen is None or (value > best if maximizing else value < best):
                best, chosen, variation = value, action, [action] + continuation
            if pruning:
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta and i + 1 < len(moves):
                    result.cutoffs += 1
                    result.skipped_children += len(moves) - i - 1
                    if trace:
                        result.events.append(dict(event="剪枝", path=path, alpha=alpha, beta=beta,
                                                  skipped=[str(x) for x in moves[i + 1:]]))
                    break
        if trace:
            # 窗口外的返回值只给出界，不能作为任意窗口下的精确缓存。
            bound = "上界" if pruning and best <= alpha_in else "下界" if pruning and best >= beta_in else "精确值"
            result.events.append(dict(event="回传", path=path, value=best, bound=bound,
                                      alpha_in=alpha_in, beta_in=beta_in, action=str(chosen)))
        return best, chosen, variation

    result.value, result.best_action, result.principal_variation = visit(initial, depth, -math.inf, math.inf, "r")
    return result


class TreeNode:
    def __init__(self, value=None, children=None, name="", is_maximizing_player=None):
        self.value = value
        self.children = list(children) if children is not None else []
        self.name = name
        self.is_maximizing_player = is_maximizing_player


def parse_tree(data, name="r"):
    if "tree_structure" in data:
        data = data["tree_structure"]
    return TreeNode(data.get("value"), [parse_tree(c, name + "/" + str(i))
                                       for i, c in enumerate(data.get("children", []))],
                    name=name, is_maximizing_player=data.get("is_maximizing_player"))


def tree_search(root, pruning=True, order="natural", seed=66, trace=False):
    def evaluate(state):
        value = state[0].value
        if value is None or not math.isfinite(value):
            raise ValueError("叶子节点必须具有有限数值得分")
        return value

    def maximizing(state):
        node, turn = state
        return turn if node.is_maximizing_player is None else node.is_maximizing_player

    def ordering(state, moves):
        if order == "natural":
            return moves
        if order == "reverse":
            return moves[::-1]
        if order == "random":
            # 随机顺序由节点名称确定，避免剪枝改变随机数消耗后影响对照。
            rng = random.Random(seed + zlib.crc32(state[0].name.encode("utf-8")))
            rng.shuffle(moves)
            return moves
        raise ValueError("节点顺序应为 natural、reverse 或 random")

    return search_game((root, True), lambda s: range(len(s[0].children)),
                       lambda s, a: (s[0].children[a], not maximizing(s)), evaluate, maximizing,
                       pruning=pruning, ordering=ordering, trace=trace)


def minimax(root):
    return tree_search(root, pruning=False).value


def alpha_beta(root):
    return tree_search(root, pruning=True).value


if __name__ == "__main__":
    import json
    import sys
    print(alpha_beta(parse_tree(json.load(sys.stdin))))
