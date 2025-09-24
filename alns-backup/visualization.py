import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
import plotly.graph_objects as go
from InputDataALNS import DataALNS


def plot_objective_changes(result, output_file_loc):
    plt.figure(figsize=(10, 6))
    result.plot_objectives(title='Changes of Objective')
    file_path = os.path.join(output_file_loc, 'images', 'Objective.svg')
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    plt.close()
    
def plot_gap_changes(tracker, output_file_loc):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(tracker.gaps) + 1), tracker.gaps)
    plt.title('Changes of Gap')
    plt.ylabel('Gap (%)')
    plt.xlabel('Iteration (#)')
    plt.grid(True, linestyle='--', alpha=0.7)
    file_path = os.path.join(output_file_loc, 'images', 'Gaps.svg')
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    plt.close()

def plot_operator_group(ax, operator_counts, title, colors):
    """辅助函数：绘制单个算子组的柱状图"""
    operator_names = [name.replace('_', ' ').title() for name in operator_counts.keys()]
    operator_counts = np.array(list(operator_counts.values()))
    cumulative_counts = operator_counts[:, :4].cumsum(axis=1)
    # 绘制堆叠条形图
    for idx in range(4):
        widths = operator_counts[:, idx]
        starts = cumulative_counts[:, idx] - widths
        bars = ax.barh(operator_names, widths, left=starts, height=0.5, 
                      color=colors[idx], label=['Best', 'Better', 'Accepted', 'Rejected'][idx])
    ax.set_title(title, pad=20, weight='bold')
    ax.set_xlabel("Iterations where operator resulted in this outcome (#)", labelpad=10)
    ax.set_ylabel("Operator", labelpad=10)
    # 添加网格线
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_operator_performance(state, result):
    # 创建图形
    plt.figure(figsize=(15, 12))
    # 创建子图网格, 2行2列, 左侧放图, 右侧放表
    gs = plt.GridSpec(2, 2, width_ratios=[1.8, 1.2], height_ratios=[1, 1])
    # 创建左侧的柱状图子图
    ax1 = plt.subplot(gs[0, 0])  # 上半部分
    ax2 = plt.subplot(gs[1, 0])  # 下半部分
    # 创建右侧的表格子图
    ax_table1 = plt.subplot(gs[0, 1])  # 上半部分的表格
    ax_table2 = plt.subplot(gs[1, 1])  # 下半部分的表格
    # 获取destroy和repair算子的统计数据
    destroy_counts = result.statistics.destroy_operator_counts
    repair_counts = result.statistics.repair_operator_counts
    # 使用科研论文风格的配色方案
    colors = ['#3C9BC9', '#76CBB4', '#FC757B', '#FFE59B']  # 淡蓝、淡绿、淡红、淡黄
    # 绘制destroy operators柱状图
    plot_operator_group(ax1, destroy_counts, "Destroy operators", colors)
    # 绘制repair operators柱状图
    plot_operator_group(ax2, repair_counts, "Repair operators", colors)
    # 创建表格数据
    destroy_data = []
    repair_data = []
    # 准备表格数据
    for op_name, counts in destroy_counts.items():
        destroy_data.append([op_name.replace('_', ' ').title()] + list(counts[:4]))  # 格式化算子名称
    for op_name, counts in repair_counts.items():
        repair_data.append([op_name.replace('_', ' ').title()] + list(counts[:4]))  # 格式化算子名称
    # 表格列标签
    columns = ['Operator', 'Best', 'Better', 'Accepted', 'Rejected']
    # 在右侧创建表格
    ax_table1.axis('off')
    ax_table2.axis('off')
    # 计算第一列的最大宽度（考虑所有行, 包括表头）
    max_width_first_col = max(
        max(len(str(row[0])) for row in destroy_data), 
        max(len(str(row[0])) for row in repair_data), 
        len(columns[0])  # 考虑表头的长度
    )
    # 计算其他列的宽度（考虑数据和表头）
    other_col_widths = []
    for i in range(1, 5):  # 对于每一列（除第一列外）
        # 获取该列所有数据的最大长度
        max_data_width = max(
            max(len(str(row[i])) for row in destroy_data), 
            max(len(str(row[i])) for row in repair_data), 
            len(columns[i])  # 考虑表头的长度
        )
        other_col_widths.append(max_data_width)
    # 设置列宽度（调整每列的相对宽度, 并添加一些额外空间）
    col_widths = [max_width_first_col/8]  # 第一列宽度, 减小除数以增加宽度
    # 为其他列添加宽度, 根据内容长度动态调整
    col_widths.extend([width/6 + 0.15 for width in other_col_widths])  # 增加其他列的宽度和间隙
    # 创建destroy operators表格
    table1 = ax_table1.table(
        cellText=destroy_data, 
        colLabels=columns, 
        loc='center', 
        cellLoc='center', 
        bbox=[0.1, 0.15, 1.1, 0.6],  # 调整表格位置和整体宽度
        colWidths=col_widths
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1.2, 1.5)
    # 设置表格样式
    for (i, j), cell in table1._cells.items():
        if i == 0:  # 表头行
            cell.set_facecolor('#E6E6E6')
            cell.set_text_props(weight='bold')
        if j == 0:  # 第一列
            cell.set_text_props(wrap=True)
    # 创建repair operators表格
    table2 = ax_table2.table(
        cellText=repair_data, 
        colLabels=columns, 
        loc='center', 
        cellLoc='center', 
        bbox=[0.1, 0.15, 1.1, 0.6],  # 调整表格位置和整体宽度
        colWidths=col_widths
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.2, 1.5)
    # 设置表格样式
    for (i, j), cell in table2._cells.items():
        if i == 0:  # 表头行
            cell.set_facecolor('#E6E6E6')
            cell.set_text_props(weight='bold')
        if j == 0:  # 第一列
            cell.set_text_props(wrap=True)
    # 添加表格标题
    ax_table1.text(0.5, 0.85, 'Destroy Operators Performance', 
                   ha='center', va='center', fontsize=10, weight='bold')
    ax_table2.text(0.5, 0.85, 'Repair Operators Performance', 
                   ha='center', va='center', fontsize=10, weight='bold')
    # 调整布局
    plt.subplots_adjust(
        left=0.1, 
        right=0.98, 
        bottom=0.2,  # 增加底部空间以容纳图例
        top=0.92,    # 减小top值, 为总标题留出更多空间
        wspace=0.3, 
        hspace=0.4
    )
    # 添加图例
    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(
        handles, 
        ['Best', 'Better', 'Accepted', 'Rejected'], 
        loc='lower center', 
        ncol=4, 
        bbox_to_anchor=(0.5, 0.05),  # 调整图例位置
        frameon=True, 
        fancybox=True, 
        shadow=True, 
        fontsize=10
    )
    # 添加总标题
    plt.suptitle(f'Performance of Operators', y=0.98, weight='bold')
    # 保存图形
    file_loc = os.path.join(state.data.output_file_loc, 'images')
    if not os.path.exists(file_loc):
        os.makedirs(file_loc)
    file_name = f'Performance of Operators.svg'
    file = os.path.join(file_loc, file_name)
    # 保存时确保所有内容都在可见区域内
    plt.savefig(file, dpi=600, bbox_inches='tight', pad_inches=0.2)
    # 关闭图形, 释放内存
    plt.close()


def plot_supply_chain_network_3d_enhanced(data: DataALNS, output_file_loc):
    """
    生成并保存一个增强的3D交互式供应链网络图。

    该函数会执行以下操作:
    1. 构建一个表示生产基地、经销商及其之间供应关系的图。
    2. 计算每个供应关系（边）上的详细信息，包括:
        - 计算生产基地在所有周期内, 可供应的SKU总数量。
        - 经销商对这些SKU的总需求量。
    3. 使用Plotly创建一个3D网络图，其中:
        - 生产基地和经销商作为节点显示。
        - 供应关系作为边显示。
        - 鼠标悬停在节点上时，显示节点的名称和类型。
        - 鼠标悬停在边上时，边会高亮显示（变为红色粗线），并显示详细的供应信息。
    4. 将生成的交互式图表保存为HTML文件。

    Args:
        data (DataALNS): 包含供应链所有输入数据的对象，
                         例如工厂、经销商、SKU、需求等。
        output_file_loc (str): 输出目录的路径，用于保存生成的HTML文件。
    """
    # 1. 构建图结构
    # 从数据对象中获取生产基地、经销商和他们之间的供应关系
    supply_chain = data.construct_supply_chain()
    G = nx.DiGraph()

    # 添加生产基地节点
    for plant in data.plants:
        G.add_node(plant, node_type='plant')

    # 添加经销商节点
    for dealer in data.dealers:
        G.add_node(dealer, node_type='dealer')

    # 添加从生产基地到经销商的边，并附带详细信息
    for (plant, dealer), skus in supply_chain.items():
        if skus:
            # 计算经销商对这些可供应SKU的总需求量
            total_demand = sum(data.demands.get((dealer, sku), 0) for sku in skus)
            
            # 计算生产基地在所有周期内, 可供应的SKU总数量
            initial_inv = sum(data.sku_initial_inv.get((plant, sku), 0) for sku in skus)
            total_prod = sum(data.sku_prod_each_day.get((plant, sku, day), 0) for sku in skus for day in range(1, data.horizons + 1))
            total_supply = initial_inv + total_prod
            
            # 添加边，并存储计算出的信息
            G.add_edge(
                plant, 
                dealer, 
                total_supply=total_supply,
                total_demand=total_demand
            )

    # 2. 设置3D布局
    # 使用spring_layout算法为图中的节点生成三维坐标
    pos = nx.spring_layout(G, dim=3, k=1.5, iterations=50, seed=42)
    node_xyz = np.array([pos[v] for v in G.nodes()])
    edge_xyz = [(pos[u], pos[v]) for u, v in G.edges()]

    # 3. 创建节点的可视化轨迹 (Traces)
    def create_node_trace(nodelist, color, symbol, size, name):
        """辅助函数，用于创建一组节点的3D散点图轨迹。"""
        idx = [list(G.nodes()).index(n) for n in nodelist]
        return go.Scatter3d(
            x=node_xyz[idx, 0], y=node_xyz[idx, 1], z=node_xyz[idx, 2],
            mode='markers',
            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=2, color='#444')),
            hoverinfo='text',
            text=[f"{n} ({'Plant' if color == '#FF6B6B' else 'Dealer'})" for n in nodelist],
            name=name
        )

    # 4. 创建边的可视化轨迹
    # 为了支持点击选择并高亮显示一条边，我们把每条边的元数据（total_supply, total_demand, endpoints）
    # 放入 trace 的 customdata 中，并确保 customdata 的长度与轨迹上的点数一致（这里为2个端点），
    # 这样在浏览器端的 Plotly 事件中可以直接读取到这些信息。
    edge_traces = []
    for (u, v), (start, end) in zip(G.edges(), edge_xyz):
        x0, y0, z0 = start
        x1, y1, z1 = end
        edge_info = G.edges[u, v]

        # 构建悬停时显示的文本信息（用于 hover）
        hover_text = (
            f"<b>{u} → {v}</b><br>"
            f"Total Supply: {edge_info['total_supply']}<br>"
            f"Total Demand: {edge_info['total_demand']}"
        )

        # customdata: 每个点都携带相同的边信息（两个端点均可访问）
        # 格式: [total_supply, total_demand, from_node, to_node]
        cd = [[edge_info['total_supply'], edge_info['total_demand'], str(u), str(v)],
              [edge_info['total_supply'], edge_info['total_demand'], str(u), str(v)]]

        # 创建每条边的轨迹；将 hovertemplate 设好以美观展示文本
        edge_traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(color='rgba(100,100,100,0.7)', width=4),
            hoverinfo='text',
            text=[hover_text, hover_text],
            customdata=cd,
            hoverlabel=dict(bgcolor="white", font_size=12),
            hovertemplate='%{text}<extra></extra>',
        ))

    # 5. 组合所有轨迹并配置图形布局
    fig = go.Figure(data=edge_traces + [
        create_node_trace(
            [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'plant'],
            '#FF6B6B', 'square', 16, 'Plant'
        ),
        create_node_trace(
            [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'dealer'],
            '#4ECDC4', 'circle', 10, 'Dealer'
        )
    ])

    # 更新布局设置
    fig.update_layout(
        title='Enhanced 3D Supply Chain Network Diagram',
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='#888', borderwidth=1),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False)
        ),
        # 增强悬停效果：当鼠标悬停在某条线上时，使其变粗变红
        hovermode='closest',
    )
    
    # 遍历所有边轨迹，设置悬停时的高亮样式
    for trace in fig.data:
        if trace.mode == 'lines':
            trace.line.width = 4 # 默认线宽
            trace.line.color = 'rgba(100,100,100,0.7)' # 默认颜色
            # Plotly本身不直接支持在Python端定义悬停样式变化，
            # 但通过清晰的hovertext和默认的hovermode='closest'，可以实现良好的交互体验。
            # 更复杂的如点击高亮需要借助Dash或者自定义JS。
            # 此处的实现是在悬停时显示详细信息，并且默认的悬停效果会使线条略微变亮。
            # 为了实现更明显的“高亮”，我们将hoverinfo做得更丰富。

    # 6. 保存为HTML文件，并在 HTML 中注入 JavaScript 以实现：
    #    - 点击一条边时高亮该边（将其颜色改为红色并加粗），
    #    - 在页面右侧显示被选中边的详细信息：total_supply 和 total_demand（以及端点信息），
    #    - 点击其它边会重置之前的高亮样式。
    #
    # 实现思路：使用 plotly 的 click 事件（plotly_click）在浏览器端处理交互。
    # 我们通过 customdata 将边信息带到前端，并使用 Plotly.restyle 来更新 trace 的样式。

    images_dir = os.path.join(output_file_loc, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    html_path = os.path.join(images_dir, 'supply_chain_network_3d_enhanced.html')

    # 将 figure 序列化为 HTML 片段（只包含图表的 div 和脚本），以便我们在其周围插入自定义 HTML/JS
    import plotly.io as pio
    fig_div = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    # 注入的 JS：
    #  - 找到图的 div（使用约定的 id 'supply_chain_network'），并监听 'plotly_click' 事件；
    #  - event.points[0].curveNumber 给出被点击 trace 的索引；
    #  - event.points[0].customdata 给出所需的边信息（我们在 customdata 中存放了 total_supply/total_demand）；
    #  - 首先恢复所有线为默认样式（灰色、宽度4），然后将被选中的 trace 置为红色并加粗；
    #  - 在页面右上角的 info 面板中显示边信息。
    post_script = f"""
    <div id='supply_chain_edge_info' style='position:fixed; right:20px; top:20px; z-index:9999; '
         'background:#ffffffcc; border:1px solid #ccc; padding:12px; border-radius:6px; '
         'box-shadow:0 4px 14px rgba(0,0,0,0.15); max-width:320px; font-family:Arial, Helvetica, sans-serif;'>
      <strong>Selected edge info</strong>
      <div id='edge_info_content' style='margin-top:8px; font-size:13px; color:#111;'>Click an edge to see details.</div>
    </div>
    <script>
    (function() {{
        // 获取 plotly 图所在的 div；to_html 生成的 div id 以 'plotly' 开头，故使用 querySelector 选择第一个 plotly 图。
        var gd = document.querySelector('[id^="plotly"]');
        if(!gd) return;

        // 默认样式值（用于重置）
        var defaultColor = 'rgba(100,100,100,0.7)';
        var defaultWidth = 4;

        // 监听点击事件
        gd.on('plotly_click', function(data) {{
            try {{
                var pt = data.points[0];
                var traceIndex = pt.curveNumber;

                // 重置所有 traces（将所有线恢复为默认样式）
                // 注意：restyle 不需要为每个 trace 单独调用，如果不指定 traceIndices 会作用于所有 traces
                Plotly.restyle(gd, {{'line.color': defaultColor, 'line.width': defaultWidth}});

                // 将被点击的 trace 高亮（红色并加粗）
                Plotly.restyle(gd, {{'line.color': 'rgba(255,0,0,0.95)', 'line.width': 8}}, [traceIndex]);

                // 读取 customdata：我们在每个边 trace 的两个点上都放了相同的 customdata
                var custom = pt.customdata || pt.data.customdata && pt.data.customdata[pt.pointIndex];
                // custom 里的格式: [total_supply, total_demand, from_node, to_node]
                var total_supply = custom[0];
                var total_demand = custom[1];
                var from_node = custom[2];
                var to_node = custom[3];

                var html = '<div><b>' + from_node + ' → ' + to_node + '</b></div>' +
                           '<div style="margin-top:6px;">Total supply (plant across all periods): <b>' + total_supply + '</b></div>' +
                           '<div style="margin-top:4px;">Total demand (dealer for these SKUs): <b>' + total_demand + '</b></div>';

                document.getElementById('edge_info_content').innerHTML = html;
            }} catch(e) {{
                console.error('Error handling plotly_click', e);
            }}
        }});

        // 当用户点击图表背景或节点，仍然希望恢复默认样式（取消选中）
        gd.on('plotly_clickannotation', function() {{
            Plotly.restyle(gd, {{'line.color': defaultColor, 'line.width': defaultWidth}});
            document.getElementById('edge_info_content').innerHTML = 'Click an edge to see details.';
        }});

        // 也提供一个双击空白处恢复样式的快捷方式
        gd.on('plotly_doubleclick', function() {{
            Plotly.restyle(gd, {{'line.color': defaultColor, 'line.width': defaultWidth}});
            document.getElementById('edge_info_content').innerHTML = 'Click an edge to see details.';
        }});
    }})();
    </script>
    """

    # 合并并写入 HTML 文件
    full_html = f"<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Supply Chain 3D Enhanced</title></head><body style=\"margin:0;\">{fig_div}{post_script}</body></html>"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
