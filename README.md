# 实验二：搜索算法

这次实验实现 A*、Minimax 和 Alpha-beta，完成迷宫、加权地形、给定博弈树和宝藏分配四题。进一步尝试同分节点排序和浅层叶子评价，观察搜索量、耗时和最终决策的变化。

## 文件

| 文件 | 内容 |
| --- | --- |
| `astar.py` | A*、BFS、启发函数、路径恢复和计数 |
| `game_search.py` | Minimax、Alpha-beta、树输入解析和剪枝记录 |
| `treasure.py` | 距离矩阵、宝藏状态、收益和评价函数 |
| `cases.py` | 课程样例和固定种子的场景生成 |
| `experiments.py` | 运行对照实验、计时和保存结果 |
| `plot_results.py` | 从结果生成图表 |
| `lab2.ipynb` | 逐题运行和解释实验过程 |
| `report/report.tex` | 五页报告源文件 |

## 环境与运行

运行环境为 Python 3.11.7。实验主体使用 Python 标准库，表格和绘图使用 pandas、NumPy 和 Matplotlib，版本见 `requirements.txt`。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python experiments.py --plots
```

在 `lab2` 目录执行命令，生成的 CSV、JSON 和图片位于 `results/`。Notebook 可以从头执行；批量实验由上面的命令单独启动。课程样例直接写在 `cases.py` 中，无需下载数据。

## 四题接口

```python
from astar import solve_maze, solve_weighted
from game_search import TreeNode, minimax, alpha_beta, parse_tree
from treasure import distance_matrix, solve_treasure

path = solve_maze(start, end, grid)
cost = solve_weighted(start, end, weighted_grid)
value = alpha_beta(root)
answer = solve_treasure(treasure_grid, values, depth=4)
```

题目一返回包含起终点的坐标元组列表，无路时返回 `[]`；题目二返回最小代价，无路时返回 `-1`。题目三返回根节点得分。题目四返回 `best_target` 和 `best_value`；`distance_matrix` 另返回名称顺序、6×6 距离字典及每对位置的路径。

也可以从标准输入读取 JSON：

```powershell
Get-Content -Raw maze.json | python astar.py maze
Get-Content -Raw weighted.json | python astar.py weighted
Get-Content -Raw tree.json | python game_search.py
Get-Content -Raw treasure.json | python treasure.py
```

寻路 JSON 包含 `start`、`end`、`grid`；博弈树接受题目中的 `tree_structure`；宝藏 JSON 包含 `grid`、`values`、`H`。Python 接口返回坐标元组，JSON 输出用坐标数组表示。

## 规则约定

- 起点初始代价为 0，每次移动支付目标格子的进入代价。坐标为 `(row, col)`，邻居顺序为右、下、左、上。
- 题目一每步代价为 1；题目二终点进入代价为 0，返回起点格时按普通代价 1 处理；题目四 A、B、所有宝藏格的进入代价均为 1。
- `X` 表示障碍。加权地图支持有限非负数；出现其他零代价格子时，启发下界会相应变弱。形状无效或负代价输入会抛出异常。
- A* 在终点从队列取出时结束。更小的 `g` 会更新前驱；关闭节点若被改善，会重新进入队列。重复堆项取出后检查是否过期。
- 题目三默认从左到右访问。示例按给出的 JSON/构造代码展开，`(6,9)` 分支被到达两次，按两次访问分别计数。
- 宝藏顺序为 A→B→A→B。只取得本回合选择的宝藏，经过其他宝藏不自动收集；玩家不挡路。负收益仍计入分数。
- 宝藏实验选择各标记互相可达的地图。若有不可达标记，接口会明确报错，场景生成记录在 `generation_attempts.json`。
- 同值行动保留当前访问顺序中首先发现的一个。根节点值用于核对正确性，多个最优行动允许不同的等价选择。

## 对照方法

寻路先固定 FIFO，比较 `h=0` 和安全下界；再固定启发函数，比较 FIFO、较大 `g` 优先、较小 `g` 优先。地图大小取 20、40、80，覆盖空图、障碍、加权地形和不可达目标，实际地图保存在 `path_cases.json`。

66、88 为场景组种子。寻路地图的随机数种子取“组种子 + 100×边长”，宝藏地图取“1000×组种子 + 候选编号”，避免不同尺寸或不同样本复用同一随机序列前缀。

题目二的安全启发为 `c_min * max(D - 1, 0)`，其中 `c_min` 是终点以外格子的最小进入代价。通用实现再加上终点的进入代价，适用于普通终点。反向 Dijkstra 提供每个节点的真实剩余代价，用于逐点检查可采纳性和逐边检查一致性。

博弈搜索比较 Minimax 与 Alpha-beta，并尝试原顺序、反顺序和随机顺序。随机顺序由种子和节点名确定，剪枝不会改变其他节点的顺序。宝藏题还加入按当前玩家即时收益降序访问的策略。

浅层评价固定为 `当前收益差 + A的最佳剩余即时收益 - B的最佳剩余即时收益`，系数为 1。种子 66、88 各生成 10 张连通地图，接受与否只看连通性。比较贪心、H=2 当前收益、H=2 机会评价和 H=4 完整搜索。每种方案选定首步后，固定该首步，再由双方完整搜索剩余行动，按同一终局收益计算差距。完整搜索的核验耗时与在线决策耗时分别处理。

## 指标与结果

寻路的 `expanded` 只计实际展开邻居的次数，终点弹出单独计入 `popped`；`pushed` 包含初始节点和重复入堆项；`peak_open` 是包含过期项的物理堆峰值。`reopened` 记录已关闭节点因更小代价重新打开的次数。

博弈的 `visited` 包含根节点和叶子，`leaves` 是实际评价的叶子数，`cutoffs` 只计确实跳过了后续子分支的事件。`skipped_children` 统计跳过的直接子分支数。

时间先预热一次，再连续计时 7 次，报告中位数和四分位距；每行 CSV 的 `time_samples_ms` 保存七次原始计时。寻路计时包括代价图构造与路径恢复；博弈计时包括排序和叶子评价。距离矩阵单独计时，博弈阶段查表使用。节点数保持确定，时间随机器负载变化。

主要结果文件：

- `given_paths.csv`、`weighted_path.csv`：前两题的路径代价和过程。
- `heuristic_audit.json`、`reopen_example.json`：启发性质与重开节点示例。
- `given_tree.csv`、`tree_traces.json`、`tree_benchmarks.csv`：剪枝与顺序实验。
- `distance_matrix.csv`、`root_values.csv`、`treasure_traces.json`：综合题的距离、首步估值和行动序列。
- `decision_benchmarks.csv`、`decision_examples.json`：浅层评价的成功与失败场景。
- `validation.json`：开发阶段独立核对的汇总，临时核对脚本已清理。

已核对规定样例、随机小地图、启发条件和博弈回传。图表均从保存结果生成，报告与 Notebook 保留改善、持平和变差的情况。

## 报告

代码仓库：[wszkxlllll/AI_Lab2](https://github.com/wszkxlllll/AI_Lab2)。姓名、学号和仓库地址在 `report/metadata.tex` 中。在报告目录运行两次 XeLaTeX：

```powershell
cd report
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

报告按五页组织，第一版保留可修改的 LaTeX 和矢量图。最终上传文件的命名与打包形式以实验二群通知为准。
