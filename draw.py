import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def generate_sample_data():
    """
    生成示例数据，模拟不同模型在不同Z3复杂度下的表现
    你可以用真实数据替换这部分
    """
    # Z3复杂度范围 (depth * branch)
    z3_complexity = np.array([1, 2, 4, 6, 8, 12, 16, 20, 25, 30, 40, 50, 60, 80])

    # 定义不同模型（你可以根据需要修改）
    # 使用学术友好且区分度高的颜色
    models = {
        'LlaMA3.2-3B-Instruct': {'color': '#E74C3C', 'marker': 'o', 'size': 6},  # 深红色
        'LlaMA3.2-8B-Instruct': {'color': '#2E86C1', 'marker': 's', 'size': 6},  # 深蓝色
        'LlaMA3.2-70B-Instruct': {'color': '#28B463', 'marker': '^', 'size': 7},  # 深绿色
        'LlaMA3.2-405B-Instruct': {'color': '#8E44AD', 'marker': 'D', 'size': 6}  # 深紫色
    }

    # 生成模拟数据 - 模拟"复杂度诅咒"现象
    # 确保大模型在相同复杂度下表现更好
    data = {}

    # 定义模型容量等级 (从小到大)
    model_capacity = {'LlaMA3.2-3B-Instruct': 0, 'LlaMA3.2-8B-Instruct': 1, 'LlaMA3.2-70B-Instruct': 2, 'LlaMA3.2-405B-Instruct': 3}

    for model_name, style in models.items():
        capacity_level = model_capacity[model_name]

        # 大模型有更高的基础性能和更好的抗复杂度能力
        base_performance = 0.85 + capacity_level * 0.05  # 基础性能: 0.85, 0.90, 0.95, 1.00
        complexity_resistance = 1.0 + capacity_level * 0.3  # 抗复杂度能力

        # 生成准确率数据，体现复杂度诅咒
        accuracy = []
        for complexity in z3_complexity:
            if complexity <= 5:
                # 简单问题：大模型表现更好
                acc = min(0.98, base_performance + np.random.normal(0, 0.02))
            elif complexity <= 15:
                # 中等复杂度：开始下降，但大模型下降更慢
                decay_factor = (complexity - 5) * (0.06 / complexity_resistance)
                acc = base_performance - decay_factor + np.random.normal(0, 0.03)
            elif complexity <= 30:
                # 高复杂度：快速下降，大模型仍有优势
                decay_factor = 0.5 + (complexity - 15) * (0.025 / complexity_resistance)
                acc = base_performance - decay_factor + np.random.normal(0, 0.05)
            else:
                # 极高复杂度：所有模型都很困难，但大模型稍好
                base_low = 0.01 + capacity_level * 0.015  # 大模型在极难情况下仍稍好
                acc = base_low + max(0, (50 - complexity) * 0.001) + np.random.normal(0, 0.01)

            accuracy.append(max(0, min(1, acc)))  # 确保在[0,1]范围内

        data[model_name] = {
            'complexity': z3_complexity,
            'accuracy': np.array(accuracy),
            'style': style
        }

    return data


def plot_z3_complexity_vs_accuracy(data, save_path=None, figsize=(12, 8)):
    """
    绘制Z3复杂度 vs 准确率图表

    参数:
    data: 包含模型数据的字典
    save_path: 保存路径（可选）
    figsize: 图表尺寸
    """

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制每个模型的曲线
    for model_name, model_data in data.items():
        style = model_data['style']
        ax.plot(model_data['complexity'],
                model_data['accuracy'] * 100,  # 转换为百分比
                color=style['color'],
                marker=style['marker'],
                markersize=style['size'],
                linewidth=2.5,
                label=model_name,
                alpha=0.8)

    # 添加"复杂度诅咒"标注区域
    curse_start = 20  # 你可以根据实际情况调整
    ax.axvline(x=curse_start, color='red', linestyle='--', alpha=0.6, linewidth=2)
    ax.text(curse_start + 2, 85, 'The curse of complexity!',
            fontsize=14, fontweight='bold', color='red',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    # 添加复杂度阈值说明
    ax.text(curse_start + 2, 75, f'Z3 Complexity > {curse_start}',
            fontsize=12, color='red', style='italic')

    # 设置坐标轴
    ax.set_xlabel('Z3 Complexity (Depth × Branch)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Logical Reasoning Performance of Different Models under Z3 Complexity', fontsize=16, fontweight='bold', pad=20)

    # 设置坐标轴范围和刻度
    ax.set_xlim(0, max([max(model_data['complexity']) for model_data in data.values()]) + 5)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 81, 10))
    ax.set_yticks(range(0, 101, 20))

    # 美化网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')

    # 图例设置
    legend = ax.legend(loc='upper right', fontsize=12, frameon=True,
                       fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"图表已保存到: {save_path}")

    return fig, ax


def load_your_data(file_path):
    """
    加载你的真实数据的函数模板

    期望的数据格式:
    CSV文件包含列: model_name, z3_complexity, accuracy
    或者
    你可以修改这个函数来适应你的数据格式
    """
    try:
        df = pd.read_csv(file_path)

        data = {}
        models = df['model_name'].unique()

        # 定义颜色和标记样式 - 学术友好且高区分度
        colors = ['#E74C3C', '#2E86C1', '#28B463', '#8E44AD', '#D35400', '#7D3C98']  # 深红、深蓝、深绿、深紫、橙色、紫色
        markers = ['o', 's', '^', 'D', 'v', '<']

        for i, model in enumerate(models):
            model_df = df[df['model_name'] == model].sort_values('z3_complexity')
            data[model] = {
                'complexity': model_df['z3_complexity'].values,
                'accuracy': model_df['accuracy'].values,
                'style': {
                    'color': colors[i % len(colors)],
                    'marker': markers[i % len(markers)],
                    'size': 6
                }
            }

        return data

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，使用示例数据")
        return generate_sample_data()
    except Exception as e:
        print(f"加载数据时出错: {e}")
        print("使用示例数据")
        return generate_sample_data()


# 主执行代码
if __name__ == "__main__":
    # 方式1: 使用示例数据
    print("生成示例数据并绘图...")
    sample_data = generate_sample_data()
    fig, ax = plot_z3_complexity_vs_accuracy(sample_data, 'z3_complexity_plot.png')
    plt.show()

    # 方式2: 使用你的真实数据（取消注释下面的代码）
    # print("加载真实数据并绘图...")
    # real_data = load_your_data('your_data.csv')  # 替换为你的数据文件路径
    # fig, ax = plot_z3_complexity_vs_accuracy(real_data, 'z3_complexity_real_data.png')
    # plt.show()

    print("绘图完成!")