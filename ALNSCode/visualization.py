"""
visualization
=============

模块定位
    提供求解过程与结果分析所需的各类可视化输出函数:
      1. 目标值变化曲线 (objective)
      2. Gap 收敛曲线
      3. 算子使用/表现统计 (堆叠水平条形图 + 表格)
      4. 供应链 3D 交互网络 (Plotly 可交互 HTML)

设计原则
    - 函数相互独立, 无全局状态
    - 输入只读 (不修改传入对象)
    - 自动创建输出目录 (images) 防止保存失败
    - Matplotlib 图在保存后即关闭, 避免内存泄漏
    - Plotly 交互图追加 JS 增强边点击高亮与侧边面板信息展示

依赖说明
    - numpy / matplotlib / networkx / plotly
    - DataALNS: 用于 3D 网络图的供应链数据构建

主要函数
    plot_objective_changes(result, output_file_loc)
        调用 result.plot_objectives() 输出目标值变化图 (依赖外部对象接口)

    plot_gap_changes(tracker, output_file_loc)
        绘制迭代 gap 收敛曲线 (tracker.gaps)

    plot_operator_performance(state, result)
        组合 4 子图 (2 组堆叠条形 + 2 组表格) 展示破坏/修复算子表现

    plot_supply_chain_network_3d_enhanced(data, output_file_loc)
        构建生产基地→经销商供应关系 3D 交互网络并生成 HTML (边点击高亮)

注意
    - 未改变原函数核心逻辑，仅:
        * 统一 imports
        * 增加目录存在性检查
        * 增补详细文档与注释
        * 去除未使用 imports (fm, Patch)
    - 已实现改进: 动态列数 / 自适应字体缩放 / Gap 平滑 (moving average 可选)
"""

# =========================
# 标准库
# =========================
import os

# =========================
# 第三方库
# =========================
import numpy as np

# =========================
# 项目内部
# =========================
from .InputDataALNS import DataALNS
from .alns_config import default_config as ALNSConfig


# ---------------------------------------------------------------------
# 基础曲线绘制
# ---------------------------------------------------------------------
def _ensure_image_dir(output_file_loc: str) -> str:
    """
    确保输出目录下的 images 子目录存在, 返回其绝对路径
    """
    img_dir = os.path.join(output_file_loc, "images")
    os.makedirs(img_dir, exist_ok=True)
    return img_dir


def plot_objective_changes(result, output_file_loc: str) -> None:
    """
    绘制目标函数变化曲线 (委托调用 result.plot_objectives).
    参数:
        result            需提供 plot_objectives(title=...) 方法的对象
        output_file_loc   输出根目录
    输出:
        保存 SVG 至 images/Objective.svg
    """
    _ensure_image_dir(output_file_loc)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    # 假设 result 已实现该接口
    result.plot_objectives(title="Changes of Objective")
    file_path = os.path.join(output_file_loc, "images", "Objective.svg")
    plt.savefig(file_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_gap_changes(tracker, output_file_loc: str, smooth_window: int = 1, show_raw: bool = True) -> None:
    """
    绘制 gap 收敛曲线 (支持可选平滑).
    参数:
        tracker         含 gaps(list[float]) 的追踪器
        output_file_loc 输出根目录
        smooth_window   平滑窗口 (>=2 启用简单移动平均); =1 表示不平滑
        show_raw        是否同时显示原始曲线 (淡色)
    输出:
        images/Gaps.svg
    行为:
        - 原始曲线: 使用 raw_color (可见) 或灰度透明 (show_raw 模式)
        - 平滑曲线: 使用 smoothed_color 与更粗线宽
        - 若 smooth_window > len(gaps) 默认降级为仅绘制原始曲线
    """
    _ensure_image_dir(output_file_loc)
    gaps = tracker.gaps
    if not gaps:
        return
    iters = range(1, len(gaps) + 1)

    # 颜色配置（明确区分 raw 与 smoothed）
    raw_color = "#d62728"        # reddish
    smoothed_color = "#1f77b4"   # blue

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    if show_raw:
        plt.plot(iters, gaps, color=raw_color, alpha=0.45, linewidth=1.2, label="Raw gap")

    if smooth_window > 1 and smooth_window <= len(gaps):
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(gaps, kernel, mode="valid")
        # 对齐 x 轴：平滑后长度减少 (smooth_window - 1)
        start = smooth_window // 2
        if len(smoothed) == len(gaps) - smooth_window + 1:
            x_sm = range(start + 1, start + 1 + len(smoothed))
        else:
            x_sm = range(1, 1 + len(smoothed))
        plt.plot(
            x_sm,
            smoothed,
            color=smoothed_color,
            linewidth=2.2,
            label=f"Smoothed (w={smooth_window})",
        )
    elif smooth_window > len(gaps):
        # 平滑窗口过大 → 退化为绘制原始曲线（若未显示过）
        if not show_raw:
            plt.plot(iters, gaps, color=raw_color, linewidth=1.6, label="Gap")

    plt.title("Changes of Gap")
    plt.ylabel("Gap (%)")
    plt.xlabel("Iteration (#)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    file_path = os.path.join(output_file_loc, "images", "Gaps.svg")
    plt.savefig(file_path, dpi=600, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# 算子表现可视化
# ---------------------------------------------------------------------
def plot_operator_group(ax, operator_counts, title: str, status_labels: list[str], colors: list[str]) -> None:
    """
    绘制单个算子组 (destroy / repair) 的堆叠水平条形图 (动态列数).
    参数:
        ax              Matplotlib Axes
        operator_counts dict[str, sequence]  每个算子对应 (s1, s2, ..., sK)
        title           子图标题
        status_labels   每列状态标签 (长度 = K)
        colors          颜色列表 (长度 >= K; 若不足需调用侧扩展)
    行为:
        - 若某算子计数序列长度 < K → 用 0 填充
        - 所有 K 列堆叠展示
    """
    operator_names = [name.replace("_", " ").title() for name in operator_counts.keys()]
    K = len(status_labels)
    rows = []
    for counts in operator_counts.values():
        c = list(counts)
        if len(c) < K:
            c += [0] * (K - len(c))
        else:
            c = c[:K]
        rows.append(c)
    counts_array = np.array(rows, dtype=float)
    cumulative = counts_array.cumsum(axis=1)

    for idx in range(K):
        widths = counts_array[:, idx]
        starts = cumulative[:, idx] - widths
        ax.barh(
            operator_names,
            widths,
            left=starts,
            height=0.5,
            color=colors[idx],
            label=status_labels[idx],
        )

    ax.set_title(title, pad=20, weight="bold")
    ax.set_xlabel("Iterations (#)", labelpad=10)
    ax.set_ylabel("Operator", labelpad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_operator_performance(state, result, cmap_name: str = "tab20") -> None:
    """
    可视化算子表现 (destroy / repair) – 动态状态列 & 自适应字体:
        - 自动检测状态列数 (基于 counts 序列长度)
        - 若状态数 > 4 扩展标签及调色
        - 行数多时自动缩小表格字体
    参数:
        state       当前解 (需含 state.data.output_file_loc)
        result      统计对象, 含 destroy_operator_counts / repair_operator_counts
        cmap_name   当需要生成额外颜色时使用的 Matplotlib 调色板
    输出:
        images/Performance of Operators.svg
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(2, 2, width_ratios=[1.8, 1.2], height_ratios=[1, 1])

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax_table1 = plt.subplot(gs[0, 1])
    ax_table2 = plt.subplot(gs[1, 1])

    destroy_counts = result.statistics.destroy_operator_counts
    repair_counts = result.statistics.repair_operator_counts

    # 确定状态列数 (取两组中最大)
    def _status_len(count_dict):
        return len(next(iter(count_dict.values()))) if count_dict else 0

    K = max(_status_len(destroy_counts), _status_len(repair_counts))
    if K == 0:
        return

    base_labels = ["Best", "Better", "Accepted", "Rejected"]
    if K <= len(base_labels):
        status_labels = base_labels[:K]
    else:
        extra = [f"S{i}" for i in range(len(base_labels) + 1, K + 1)]
        status_labels = base_labels + extra

    # 颜色生成
    base_colors = ["#3C9BC9", "#76CBB4", "#FC757B", "#FFE59B"]
    if K <= len(base_colors):
        colors = base_colors[:K]
    else:
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i % cmap.N) for i in range(K)]

    # 绘制堆叠条形
    plot_operator_group(ax1, destroy_counts, "Destroy operators", status_labels, colors)
    plot_operator_group(ax2, repair_counts, "Repair operators", status_labels, colors)

    # 表格数据准备 (补零对齐)
    def _pad_counts(counts_list, K):
        c = list(counts_list)
        if len(c) < K:
            c += [0] * (K - len(c))
        else:
            c = c[:K]
        return c

    destroy_data = [
        [op_name.replace("_", " ").title()] + _pad_counts(counts, K)
        for op_name, counts in destroy_counts.items()
    ]
    repair_data = [
        [op_name.replace("_", " ").title()] + _pad_counts(counts, K)
        for op_name, counts in repair_counts.items()
    ]
    columns = ["Operator"] + status_labels

    ax_table1.axis("off")
    ax_table2.axis("off")

    # 动态列宽
    max_width_first_col = max(
        [len(str(row[0])) for row in destroy_data] + [len(str(row[0])) for row in repair_data] + [len(columns[0])]
    )
    other_col_widths = []
    for i in range(1, K + 1):
        other_col_widths.append(
            max(
                [len(str(row[i])) for row in destroy_data]
                + [len(str(row[i])) for row in repair_data]
                + [len(columns[i])]
            )
        )
    col_widths = [max_width_first_col / 8]
    col_widths.extend([w / 6 + 0.15 for w in other_col_widths])

    # 字体自适应
    max_rows = max(len(destroy_data), len(repair_data))
    if max_rows <= 10:
        fsize = 9
    elif max_rows <= 20:
        fsize = 8
    elif max_rows <= 30:
        fsize = 7
    else:
        fsize = 6

    def _build_table(ax_table, data):
        tbl = ax_table.table(
            cellText=data,
            colLabels=columns,
            loc="center",
            cellLoc="center",
            bbox=[0.1, 0.15, 1.1, 0.6],
            colWidths=col_widths,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fsize)
        # scale 依据状态列数适度压缩
        scale_y = 1.2 if max_rows < 18 else 1.0
        tbl.scale(1.0, scale_y)
        for (i, j), cell in tbl._cells.items():
            if i == 0:
                cell.set_facecolor("#E6E6E6")
                cell.set_text_props(weight="bold")
            if j == 0:
                cell.set_text_props(wrap=True)
        return tbl

    _build_table(ax_table1, destroy_data)
    _build_table(ax_table2, repair_data)

    ax_table1.text(
        0.5, 0.85, "Destroy Operators Performance", ha="center", va="center", fontsize=10, weight="bold"
    )
    ax_table2.text(
        0.5, 0.85, "Repair Operators Performance", ha="center", va="center", fontsize=10, weight="bold"
    )

    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.92, wspace=0.3, hspace=0.4)

    # 图例 (所有状态)
    handles, _ = ax1.get_legend_handles_labels()
    plt.figlegend(
        handles,
        status_labels,
        loc="lower center",
        ncol=min(K, 6),
        bbox_to_anchor=(0.5, 0.05),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )

    plt.suptitle("Performance of Operators", y=0.98, weight="bold")

    file_loc = _ensure_image_dir(state.data.output_file_loc)
    file = os.path.join(file_loc, "Performance of Operators.svg")
    plt.savefig(file, dpi=600, bbox_inches="tight", pad_inches=0.2)
    plt.close()


# ---------------------------------------------------------------------
# 供应链 3D 交互网络
# ---------------------------------------------------------------------
def plot_supply_chain_network_3d_enhanced(data: DataALNS, output_file_loc: str) -> None:
    """
    构建生产基地→经销商供应关系 3D 交互网络 (Plotly).
    节点: plant / dealer
    边属性: total_supply (plant 可供, 初始库存 + 生产), total_demand (dealer 对交集 SKU 需求)
    交互:
      - 悬停显示边指标
      - 点击边高亮 (红色/加粗) 并在右上信息框展示详情
      - 双击 / 点击背景 重置
    输出:
        images/supply_chain_network_3d_enhanced.html
    """
    # 1. 图构建
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.io as pio

    supply_chain = data.construct_supply_chain()
    G = nx.DiGraph()
    for plant in data.plants:
        G.add_node(plant, node_type="plant")
    for dealer in data.dealers:
        G.add_node(dealer, node_type="dealer")

    for (plant, dealer), skus in supply_chain.items():
        if skus:
            total_demand = sum(data.demands.get((dealer, sku), 0) for sku in skus)
            initial_inv = sum(data.sku_initial_inv.get((plant, sku), 0) for sku in skus)
            total_prod = sum(
                data.sku_prod_each_day.get((plant, sku, day), 0)
                for sku in skus
                for day in range(1, data.horizons + 1)
            )
            total_supply = initial_inv + total_prod
            G.add_edge(plant, dealer, total_supply=total_supply, total_demand=total_demand)

    # 2. 布局 (spring 3D)
    pos = nx.spring_layout(G, dim=3, k=1.5, iterations=50, seed=42)
    node_xyz = np.array([pos[v] for v in G.nodes()])
    edge_xyz = [(pos[u], pos[v]) for u, v in G.edges()]

    def create_node_trace(nodelist, color, symbol, size, name):
        """
        构造节点散点轨迹
        color 决定节点类型 (用于 hover 文本)
        """
        idx = [list(G.nodes()).index(n) for n in nodelist]
        return go.Scatter3d(
            x=node_xyz[idx, 0],
            y=node_xyz[idx, 1],
            z=node_xyz[idx, 2],
            mode="markers",
            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=2, color="#444")),
            hoverinfo="text",
            text=[f"{n} ({'Plant' if color == '#FF6B6B' else 'Dealer'})" for n in nodelist],
            name=name,
        )

    # 3. 边轨迹
    edge_traces = []
    for (u, v), (start, end) in zip(G.edges(), edge_xyz):
        x0, y0, z0 = start
        x1, y1, z1 = end
        edge_info = G.edges[u, v]
        hover_text = (
            f"<b>{u} → {v}</b><br>"
            f"Total Supply: {edge_info['total_supply']}<br>"
            f"Total Demand: {edge_info['total_demand']}"
        )
        custom = [
            [
                edge_info["total_supply"],
                edge_info["total_demand"],
                str(u),
                str(v),
            ],
            [
                edge_info["total_supply"],
                edge_info["total_demand"],
                str(u),
                str(v),
            ],
        ]
        edge_traces.append(
            go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode="lines",
                line=dict(color="rgba(100,100,100,0.7)", width=4),
                hoverinfo="text",
                text=[hover_text, hover_text],
                customdata=custom,
                hoverlabel=dict(bgcolor="white", font_size=12),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # 4. 组合图形
    fig = go.Figure(
        data=edge_traces
        + [
            create_node_trace(
                [n for n, attr in G.nodes(data=True) if attr["node_type"] == "plant"],
                "#FF6B6B",
                "square",
                16,
                "Plant",
            ),
            create_node_trace(
                [n for n, attr in G.nodes(data=True) if attr["node_type"] == "dealer"],
                "#4ECDC4",
                "circle",
                10,
                "Dealer",
            ),
        ]
    )

    fig.update_layout(
        title="Enhanced 3D Supply Chain Network Diagram",
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#888",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        ),
        hovermode="closest",
    )

    # 5. 保存 HTML + 注入交互 JS
    images_dir = _ensure_image_dir(output_file_loc)
    html_path = os.path.join(images_dir, "supply_chain_network_3d_enhanced.html")

    fig_div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    post_script = """
    <div id='supply_chain_edge_info' style='position:fixed; right:20px; top:20px; z-index:9999;
         background:#ffffffcc; border:1px solid #ccc; padding:12px; border-radius:6px;
         box-shadow:0 4px 14px rgba(0,0,0,0.15); max-width:320px; font-family:Arial, Helvetica, sans-serif;'>
      <strong>Selected edge info</strong>
      <div id='edge_info_content' style='margin-top:8px; font-size:13px; color:#111;'>Click an edge to see details.</div>
    </div>
    <script>
    (function() {
        var gd = document.querySelector('[id^="plotly"]');
        if(!gd) return;
        var defaultColor = 'rgba(100,100,100,0.7)';
        var defaultWidth = 4;

        gd.on('plotly_click', function(data) {
            try {
                var pt = data.points[0];
                var traceIndex = pt.curveNumber;

                Plotly.restyle(gd, {'line.color': defaultColor, 'line.width': defaultWidth});
                Plotly.restyle(gd, {'line.color': 'rgba(255,0,0,0.95)', 'line.width': 8}, [traceIndex]);

                var custom = pt.customdata || (pt.data.customdata && pt.data.customdata[pt.pointIndex]);
                var total_supply = custom[0];
                var total_demand = custom[1];
                var from_node = custom[2];
                var to_node = custom[3];

                var html = '<div><b>' + from_node + ' → ' + to_node + '</b></div>' +
                           '<div style="margin-top:6px;">Total supply (plant across all periods): <b>' + total_supply + '</b></div>' +
                           '<div style="margin-top:4px;">Total demand (dealer for these SKUs): <b>' + total_demand + '</b></div>';
                document.getElementById('edge_info_content').innerHTML = html;
            } catch(e) {
                console.error('Error handling plotly_click', e);
            }
        });

        function resetEdgeHighlight() {
            Plotly.restyle(gd, {'line.color': defaultColor, 'line.width': defaultWidth});
            document.getElementById('edge_info_content').innerHTML = 'Click an edge to see details.';
        }

        gd.on('plotly_clickannotation', resetEdgeHighlight);
        gd.on('plotly_doubleclick', resetEdgeHighlight);
    })();
    </script>
    """

    full_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Supply Chain 3D Enhanced</title></head>"
        "<body style='margin:0;'>"
        f"{fig_div}{post_script}"
        "</body></html>"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
