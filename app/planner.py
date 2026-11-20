"""分块尺寸选择：在激活预算内选 1..64 中最大的方形输出块边长。

预算口径：
  对每个输出分块（边缘块按实际输出裁短），沿两层感受野链路反推出覆盖该块
  所需的 *图内* 激活最小矩形窗口，元素数 = 窗口高 × 窗口宽 × 输入通道数，
  取全网格所有分块中的最大值作为该边长下的峰值用量。图外零填充不物化、
  不计入。实跑时 TiledRunner 截取的正是同一个窗口，因此运行期峰值与此处
  预算一致。

窗口链路：layer2 dilation2 使输出 i 读 layer1 输出 {i-2, i, i+2}（裁到有效
输出界）；layer1 stride2 再映射到输入（见 conv.required_input_window）。
行窗口只取决于行方向、列窗口只取决于列方向，故峰值可在两个轴上分别求
最大值再相乘，无需枚举全网格的每个块对。
"""

from __future__ import annotations

from dataclasses import dataclass

from .conv import (
    L1_IN_CH,
    MAX_TILE_SIDE,
    input_span,
    layer1_out_size,
    layer1_output_span,
)

@dataclass
class TilePlan:
    feasible: bool
    tile: int | None            # 选中的方形边长（不可行时为 None）
    budget: int                 # 用户预算（元素数）
    minimum_budget: int         # s=1 时所需的最低预算
    peak_input_elems: int       # 选中边长下的实际峰值图内激活元素数
    tile_grid: tuple[int, int]  # (行块数, 列块数)
    per_side_peaks: list[tuple[int, int]]  # 各候选边长 -> 峰值，供页面展示
    output_size: tuple[int, int]


def _peak_for_side(h: int, w: int, ho: int, wo: int, s: int) -> int:
    """边长 s 下，所有分块（含边缘裁短块）所需图内激活元素数的最大值。

    峰值 = (行方向最大窗口高) × (列方向最大窗口宽) × 输入通道数。
    这里遍历该轴上的每个分块起点，取得精确最大值；边缘块按实际输出尺寸
    裁短后单独计算，因此不会漏掉“短边块反而窗口更大”的情形。
    """
    max_h = 0
    for r0 in range(0, ho, s):
        kmin, kmax = layer1_output_span(r0, min(r0 + s, ho), ho)
        ir0, ir1 = input_span(kmin, kmax, h)
        max_h = max(max_h, ir1 - ir0)
    max_w = 0
    for c0 in range(0, wo, s):
        jmin, jmax = layer1_output_span(c0, min(c0 + s, wo), wo)
        ic0, ic1 = input_span(jmin, jmax, w)
        max_w = max(max_w, ic1 - ic0)
    return max_h * max_w * L1_IN_CH


def plan_tiles(h: int, w: int, budget: int) -> TilePlan:
    if budget < 1:
        raise ValueError("激活预算必须是正整数（元素数）")
    ho, wo = layer1_out_size(h, w)

    # 候选边长不超过输出本身的尺寸，也不超过 64
    upper = min(MAX_TILE_SIDE, ho, wo)
    per_side: list[tuple[int, int]] = []
    side_peaks: dict[int, int] = {}
    for s in range(1, upper + 1):
        p = _peak_for_side(h, w, ho, wo, s)
        side_peaks[s] = p
        per_side.append((s, p))

    minimum_budget = side_peaks[1]
    if budget < minimum_budget:
        # 单像素块仍超限：不执行推理，只报最低预算
        return TilePlan(
            feasible=False, tile=None, budget=budget,
            minimum_budget=minimum_budget, peak_input_elems=minimum_budget,
            tile_grid=(ho, wo), per_side_peaks=per_side, output_size=(ho, wo),
        )

    # 从大到小选预算内最大的方形边长
    chosen = next(s for s in range(upper, 0, -1) if side_peaks[s] <= budget)
    grid = ((ho + chosen - 1) // chosen, (wo + chosen - 1) // chosen)
    return TilePlan(
        feasible=True, tile=chosen, budget=budget,
        minimum_budget=minimum_budget,
        peak_input_elems=side_peaks[chosen],
        tile_grid=grid, per_side_peaks=per_side, output_size=(ho, wo),
    )
