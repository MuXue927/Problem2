import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# # Set Chinese font
# plt.rcParams['font.family'] = 'SimSun'  # 使用系统宋体
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 针对sans-serif字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Data
labels = ['Growth of followers', 'Likes', 'Comments', 'New customers', 'Orders']
target = [125, 150, 40, 100, 30]
actual = [106, 190, 15, 48, 24]
achievement = [a/t * 100 for a, t in zip(actual, target)]

# Radar chart setup
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
achievement += achievement[:1]
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, achievement, color='#87CEEB', alpha=0.5)
ax.plot(angles, achievement, color='#FFD700', linewidth=2.5)
ax.set_yticks([0, 50, 100, 150])
ax.set_yticklabels(['0%', '50%', '100%', '150%'], fontsize=10)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, weight='bold')
plt.title('KPI Achievement Rate', fontsize=16, weight='bold', pad=20)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('images/KPI达成率.svg', dpi=600)