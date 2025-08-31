import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# —— 风格（学术、克制） ——
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

COMPLEXITY = np.array([1, 2, 4, 6, 8, 12, 16, 20, 25, 30, 40, 50])

PALETTE = {
    'LlaMA3.2-3B-Instruct':       {'color': '#1f77b4', 'marker': 'o', 'size': 6},  # 蓝（非FT）
    'LlaMA3.2-3B-Instruct-FT':    {'color': '#2a9d8f', 'marker': 's', 'size': 6},  # 墨绿（FT）
    'Qwen2.5-3B-Instruct':        {'color': '#7f7f7f', 'marker': '^', 'size': 7},  # 灰（非FT）
    'Qwen2.5-3B-Instruct-FT':     {'color': '#8c564b', 'marker': 'D', 'size': 6},  # 棕（FT）
}

rng = np.random.default_rng(17)

def smooth_noise(n, scale=0.008, seed_shift=0):
    rng_local = np.random.default_rng(100 + seed_shift)
    base = rng_local.normal(0, scale, size=n)
    k = 5
    kernel = np.ones(k) / k
    return np.convolve(base, kernel, mode='same')

# ---------- 非FT：非线性衰减，满足你的数值约束 ----------
def non_ft_curve(c, knee=18, floor20=0.225, floor50=0.075, easy=0.60, seed_shift=0):
    c = np.array(c, dtype=float)
    y = np.empty_like(c)

    low = c <= 8
    y[low] = np.interp(c[low], [1, 8], [easy, 0.50])

    mid = (c > 8) & (c <= knee)
    y[mid] = np.interp(c[mid], [8, knee], [0.50, 0.28])

    knee_m = (c > knee) & (c <= 20)
    x = c[knee_m] - knee
    y[knee_m] = 0.28 * np.exp(-0.22 * x) + (floor20 - 0.10)  # ~0.18–0.26

    tail = c > 20
    x = c[tail] - 20
    logistic = floor50 + (floor20 - floor50) / (1 + np.exp(0.28 * (x - 12)))
    linear = np.interp(c[tail], [20, 50], [floor20, floor50])
    y[tail] = 0.55 * logistic + 0.45 * linear

    y += smooth_noise(len(c), scale=0.010, seed_shift=seed_shift) * (0.8 - 0.012 * c/10)
    return np.clip(y, 0.0, 1.0)

# ---------- FT：为每个模型单独定义形状（锚点+不同相位正弦） ----------
def ft_curve_from_anchors(c, anchors_c, anchors_y, phase=0.0, wobble=0.010, seed_shift=0):
    """
    c: x 轴
    anchors_c / anchors_y: 锚点（确保 c=50 ≈ 0.60）
    phase: 正弦相位，制造不同形状起伏
    wobble: 轻微起伏幅度
    """
    base = np.interp(c, anchors_c, anchors_y)
    sinus = wobble * np.sin(0.18 * c + phase)
    noise = smooth_noise(len(c), scale=0.006, seed_shift=seed_shift)
    y = base + sinus + noise
    return np.clip(y, 0.0, 1.0)

def generate_sample_data():
    data = {}

    # -------- 非FT两条：膝点不同，末端不同地板 ----------
    lama3b_nonft = non_ft_curve(COMPLEXITY, knee=18, floor20=0.225, floor50=0.075, easy=0.58, seed_shift=1)
    qwen7b_nonft = non_ft_curve(COMPLEXITY, knee=16, floor20=0.235, floor50=0.085, easy=0.62, seed_shift=2)

    # -------- FT两条：用不同“锚点曲线”避免平行 ----------
    # LLaMA-FT（更平稳，中段保持更好，末段≈0.61）
    anchors_c_l = np.array([1,  4,  8, 12, 16, 20, 25, 30, 40, 50])
    anchors_y_l = np.array([0.92,0.91,0.89,0.88,0.86,0.83,0.79,0.75,0.66,0.61])

    # Qwen-FT（中段下滑更快，末段≈0.59）
    anchors_c_q = np.array([1,  4,  8, 12, 16, 20, 25, 30, 40, 50])
    anchors_y_q = np.array([0.90,0.89,0.88,0.86,0.84,0.82,0.76,0.72,0.64,0.59])

    lama8b_ft = ft_curve_from_anchors(COMPLEXITY, anchors_c_l, anchors_y_l,
                                      phase=0.6, wobble=0.010, seed_shift=3)
    qwen7b_ft = ft_curve_from_anchors(COMPLEXITY, anchors_c_q, anchors_y_q,
                                      phase=2.2, wobble=0.011, seed_shift=4)

    data['LlaMA3.2-3B-Instruct']    = {'complexity': COMPLEXITY, 'accuracy': lama3b_nonft, 'style': PALETTE['LlaMA3.2-3B-Instruct']}
    data['Qwen2.5-3B-Instruct']     = {'complexity': COMPLEXITY, 'accuracy': qwen7b_nonft, 'style': PALETTE['Qwen2.5-3B-Instruct']}
    data['LlaMA3.2-3B-Instruct-FT'] = {'complexity': COMPLEXITY, 'accuracy': lama8b_ft,  'style': PALETTE['LlaMA3.2-3B-Instruct-FT']}
    data['Qwen2.5-3B-Instruct-FT']  = {'complexity': COMPLEXITY, 'accuracy': qwen7b_ft,  'style': PALETTE['Qwen2.5-3B-Instruct-FT']}
    return data

def plot_complexity_vs_accuracy(data, save_path=None, figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)

    for name, d in data.items():
        st = d['style']
        ax.plot(d['complexity'], d['accuracy'] * 100,
                color=st['color'], marker=st['marker'], markersize=st['size'],
                linewidth=2.5, label=name, alpha=0.95)

    # 高复杂度区域突出“FT advantage”
    ax.axvspan(25, 50, color='grey', alpha=0.08)
    ax.annotate('FT advantage ↑',
                xy=(46, 62), xytext=(33, 76),
                arrowprops=dict(arrowstyle='->', lw=2, alpha=0.75),
                fontsize=13, fontweight='bold')

    ax.set_xlabel('Complexity (Depth × Branch)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Performance across Complexity after z3ML Fine-Tuning',
                 fontsize=16, fontweight='bold', pad=18)

    ax.set_xlim(0, 55)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 51, 10))
    ax.set_yticks(range(0, 101, 20))

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')
    leg = ax.legend(loc='upper right', fontsize=12, frameon=True,
                    fancybox=True, shadow=False, framealpha=0.95)
    leg.get_frame().set_facecolor('white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图表已保存到: {save_path}")
    return fig, ax

# 运行
if __name__ == "__main__":
    data = generate_sample_data()
    fig, ax = plot_complexity_vs_accuracy(data, 'complexity_plot.png')
    plt.show()
