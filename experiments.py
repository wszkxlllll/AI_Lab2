"""运行课程样例、自设对照和决策质量实验。"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import time

import pandas as pd

from astar import (adjacent, astar_graph, astar_grid, bfs_grid, distances_to_goal,
                   make_heuristic, terrain_costs)
from cases import (MAZE, WEIGHTED, TREASURE_MAP, TREASURE_VALUES,
                   example_tree, path_scenarios, random_tree, treasure_candidate)
from game_search import tree_search
from treasure import TreasureGame, distance_matrix


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPEATS = 7


def plain(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save_json(name, value):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def timed(call):
    """先预热一次，返回七次计时的中位数与四分位距。"""
    result = call()
    timings = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        result = call()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    q1, _, q3 = statistics.quantiles(timings, n=4, method="inclusive")
    return result, dict(median_ms=statistics.median(timings), iqr_ms=q3 - q1,
                        repeats=REPEATS, samples_ms=timings)


def path_row(result, timing, **metadata):
    return dict(**metadata, cost=result.cost if math.isfinite(result.cost) else None,
                reachable=bool(result.path), path_steps=max(len(result.path) - 1, 0),
                expanded=result.expanded, pushed=result.pushed, popped=result.popped,
                stale=result.stale, reopened=result.reopened, peak_open=result.peak_open,
                time_samples_ms=json.dumps(timing["samples_ms"]),
                **{k: v for k, v in timing.items() if k != "samples_ms"})


def game_row(result, timing, **metadata):
    return dict(**metadata, value=result.value, first=result.best_action,
                visited=result.visited, leaves=result.leaves, cutoffs=result.cutoffs,
                skipped_children=result.skipped_children,
                time_samples_ms=json.dumps(timing["samples_ms"]),
                **{k: v for k, v in timing.items() if k != "samples_ms"})


def run_given_cases():
    path_records, trace_records = [], {}
    for task, grid, goal, weighted, free in [
        ("q1", MAZE, (3, 4), False, False), ("q2", WEIGHTED, (4, 4), True, True)
    ]:
        for mode, tie in [("zero", "fifo"), ("manhattan", "fifo"), ("safe", "fifo"), ("safe", "larger_g")]:
            call = lambda: astar_grid((0, 0), goal, grid, weighted, free, mode, tie)
            result, timing = timed(call)
            path_records.append(path_row(result, timing, task=task, heuristic=mode, tie=tie))
            recorded = astar_grid((0, 0), goal, grid, weighted, free, mode, tie, trace=True)
            trace_records[f"{task}_{mode}_{tie}"] = asdict(recorded)
        if task == "q1":
            result, timing = timed(lambda: bfs_grid((0, 0), goal, grid))
            path_records.append(path_row(result, timing, task=task, heuristic="bfs", tie="fifo"))
    pd.DataFrame(path_records).to_csv(RESULTS / "given_paths.csv", index=False)
    save_json("path_traces.json", trace_records)
    costs = terrain_costs(WEIGHTED, True, (4, 4), True)
    path = astar_grid((0, 0), (4, 4), WEIGHTED, True, True).path
    cumulative, rows = 0, []
    for i, node in enumerate(path):
        step = 0 if i == 0 else costs[node]
        cumulative += step
        rows.append(dict(step=i, row=node[0], col=node[1], entered_cost=step, cumulative=cumulative))
    pd.DataFrame(rows).to_csv(RESULTS / "weighted_path.csv", index=False)

    exact = distances_to_goal(costs, (4, 4))
    audit = []
    for mode in ("zero", "manhattan", "safe"):
        h = make_heuristic(costs, (4, 4), mode)
        over = [dict(node=n, estimate=h(n), exact=exact[n]) for n in costs if n in exact and h(n) > exact[n]]
        inconsistent = [dict(source=n, target=v, h_source=h(n), edge=w, h_target=h(v))
                        for n in costs for v, w in adjacent(n, costs) if h(n) > w + h(v)]
        audit.append(dict(heuristic=mode, admissibility_violations=len(over),
                          consistency_violations=len(inconsistent), examples=over, edges=inconsistent))
    save_json("heuristic_audit.json", audit)

    graph = {"S": [("A", 3), ("B", 1)], "A": [("G", 3)], "B": [("A", 1)], "G": []}
    h = {"S": 0, "A": 0, "B": 4, "G": 0}
    save_json("reopen_example.json", {str(reopen): asdict(astar_graph(
        "S", "G", lambda n: graph[n], lambda n: h[n], reopen=reopen, trace=True)) for reopen in (False, True)})

    tree_records, tree_traces = [], {}
    tree = example_tree()
    for pruning in (False, True):
        for order in ("natural", "reverse", "random"):
            result, timing = timed(lambda: tree_search(tree, pruning, order, seed=66))
            tree_records.append(game_row(result, timing, algorithm="alpha_beta" if pruning else "minimax", order=order))
            tree_traces[f"{pruning}_{order}"] = asdict(tree_search(example_tree(), pruning, order, seed=66, trace=True))
    pd.DataFrame(tree_records).to_csv(RESULTS / "given_tree.csv", index=False)
    save_json("tree_traces.json", tree_traces)

    (names, distances, details), timing = timed(lambda: distance_matrix(TREASURE_MAP))
    pd.DataFrame(distances).T.loc[names, names].to_csv(RESULTS / "distance_matrix.csv")
    save_json("distance_paths.json", details)
    save_json("distance_timing.json", timing)
    game = TreasureGame(distances, TREASURE_VALUES)
    rows, traces, root_values = [], {}, []
    for depth in (2, 4):
        for pruning in (False, True):
            for order in ("natural", "reverse", "gain"):
                result, timing = timed(lambda: game.solve(depth, pruning=pruning, order=order))
                rows.append(game_row(result, timing, depth=depth, algorithm="alpha_beta" if pruning else "minimax", order=order))
                if pruning and order == "natural":
                    traces[str(depth)] = dict(result=asdict(game.solve(depth, trace=True)),
                                             trajectory=game.trajectory(result.principal_variation))
        for target in game.initial.remaining:
            next_state = game.step(game.initial, target)
            result = game.solve(depth - 1, pruning=False, state=next_state)
            root_values.append(dict(depth=depth, target=target, value=result.value,
                                    terminal_value=game.complete_after(target)))
    pd.DataFrame(rows).to_csv(RESULTS / "given_treasure.csv", index=False)
    pd.DataFrame(root_values).to_csv(RESULTS / "root_values.csv", index=False)
    save_json("treasure_traces.json", traces)
    assert game.solve(2).best_action == "T2" and game.solve(2).value == -1
    assert game.solve(4).best_action == "T3" and game.solve(4).value == 3
    return game


def run_path_benchmarks():
    rows, saved = [], []
    for case in path_scenarios():
        saved.append(case)
        goal = (case["size"] - 1, case["size"] - 1)
        reference = astar_grid((0, 0), goal, case["grid"], weighted=case["weighted"], heuristic="zero")
        for mode, tie in [("zero", "fifo"), ("safe", "fifo"), ("safe", "larger_g"), ("safe", "smaller_g")]:
            result, timing = timed(lambda: astar_grid((0, 0), goal, case["grid"], weighted=case["weighted"], heuristic=mode, tie=tie))
            assert result.cost == reference.cost
            rows.append(path_row(result, timing, name=case["name"], family=case["family"],
                                 size=case["size"], seed=case["seed"], heuristic=mode, tie=tie))
    pd.DataFrame(rows).to_csv(RESULTS / "path_benchmarks.csv", index=False)
    save_json("path_cases.json", saved)


def run_tree_benchmarks():
    rows = []
    for seed in (66, 88):
        for branching, depth in ((2, 4), (2, 6), (2, 8), (3, 4), (3, 6), (4, 5)):
            tree = random_tree(branching, depth, seed)
            truth = tree_search(tree, pruning=False).value
            for pruning in (False, True):
                for order in ("natural", "reverse", "random"):
                    result, timing = timed(lambda: tree_search(tree, pruning, order, seed))
                    assert result.value == truth
                    rows.append(game_row(result, timing, seed=seed, branching=branching, depth=depth,
                                         algorithm="alpha_beta" if pruning else "minimax", order=order))
    pd.DataFrame(rows).to_csv(RESULTS / "tree_benchmarks.csv", index=False)


def decision_rows(game, name, seed):
    full = game.solve(4, pruning=False)
    rows = []
    for method, depth, evaluation in (("H2_current", 2, "current"),
                                       ("H2_opportunity", 2, "opportunity"),
                                       ("H4_full", 4, "current")):
        result, timing = timed(lambda: game.solve(depth, evaluation=evaluation, order="natural"))
        terminal = game.complete_after(result.best_action)
        rows.append(game_row(result, timing, name=name, seed=seed, method=method,
                             terminal_value=terminal, optimal_value=full.value,
                             regret=full.value - terminal, optimal_first=(terminal == full.value)))
    first, timing = timed(lambda: max(game.initial.remaining, key=lambda t: game.gain(game.initial, t)))
    terminal = game.complete_after(first)
    rows.append(dict(name=name, seed=seed, method="greedy", first=first,
                     value=game.gain(game.initial, first), visited=None, leaves=None, cutoffs=0,
                     action_evaluations=len(game.initial.remaining), skipped_children=0,
                     time_samples_ms=json.dumps(timing["samples_ms"]),
                     **{k: v for k, v in timing.items() if k != "samples_ms"},
                     terminal_value=terminal, optimal_value=full.value,
                     regret=full.value - terminal, optimal_first=(terminal == full.value)))
    return rows


def run_decision_benchmarks(given):
    rows = decision_rows(given, "given", None)
    scenarios, attempts = [], []
    for seed in (66, 88):
        accepted, index = 0, 0
        while accepted < 10:
            grid, values = treasure_candidate(seed, index)
            _, distances, _ = distance_matrix(grid)
            connected = all(math.isfinite(v) for row in distances.values() for v in row.values())
            attempts.append(dict(seed=seed, candidate=index, connected=connected))
            if connected:
                name = f"random_{seed}_{index}"
                game = TreasureGame(distances, values)
                rows.extend(decision_rows(game, name, seed))
                scenarios.append(dict(name=name, seed=seed, candidate=index, grid=grid,
                                      values=values, distances=distances))
                accepted += 1
            index += 1
            if index > 1000:
                raise RuntimeError("生成连通宝藏场景失败")
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "decision_benchmarks.csv", index=False)
    frame[frame.name != "given"].groupby(["seed", "method"]).agg(
        cases=("name", "count"), optimal_first_rate=("optimal_first", "mean"),
        mean_regret=("regret", "mean"), max_regret=("regret", "max"),
        mean_visited=("visited", "mean"), median_ms=("median_ms", "median")
    ).to_csv(RESULTS / "decision_summary.csv")
    save_json("treasure_cases.json", scenarios)
    save_json("generation_attempts.json", attempts)
    explanations = []
    for case in scenarios:
        game = TreasureGame(case["distances"], case["values"])
        roots = []
        for target in game.initial.remaining:
            state = game.step(game.initial, target)
            roots.append(dict(target=target, immediate_gain=game.gain(game.initial, target),
                              current_h2=game.solve(1, state=state, pruning=False).value,
                              opportunity_h2=game.solve(1, state=state, evaluation="opportunity", pruning=False).value,
                              full_h4=game.complete_after(target)))
        current = game.solve(2).best_action
        improved = game.solve(2, evaluation="opportunity").best_action
        if current != improved:
            explanations.append(dict(name=case["name"], values=case["values"], roots=roots,
                                     current_first=current, opportunity_first=improved,
                                     current_trajectory=game.trajectory([current] + game.solve(
                                         3, state=game.step(game.initial, current)).principal_variation),
                                     opportunity_trajectory=game.trajectory([improved] + game.solve(
                                         3, state=game.step(game.initial, improved)).principal_variation)))
    save_json("decision_examples.json", explanations)


def main():
    parser = argparse.ArgumentParser(description="运行全部实验并保存结果")
    parser.add_argument("--plots", action="store_true", help="同时生成图表")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    start = time.perf_counter()
    game = run_given_cases()
    print("规定样例完成", flush=True)
    run_path_benchmarks()
    print("寻路对照完成", flush=True)
    run_tree_benchmarks()
    print("剪枝对照完成", flush=True)
    run_decision_benchmarks(game)
    print("决策质量对照完成", flush=True)
    save_json("environment.json", dict(python=platform.python_version(), platform=platform.platform(),
                                       repeats=REPEATS, seeds=[66, 88], wall_seconds=time.perf_counter() - start,
                                       source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                      for p in ROOT.glob("*.py") if not p.name.startswith("_")}))
    if args.plots:
        from plot_results import main as plots
        plots()


if __name__ == "__main__":
    main()
