import os
"""
Professional pipeline diagram for scientific paper
Clean, modern design following academic standards
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import paths  # noqa: F401  # TSC path config

# Create figure with better proportions
fig = plt.figure(figsize=(20, 4.5), facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(-0.5, 19.5)
ax.set_ylim(-0.2, 4.2)
ax.axis('off')

# Professional color scheme (subtle, academic)
colors = {
    'data': '#D6EAF8',       # Soft blue
    'transform': '#FADBD8',  # Soft coral
    'jm': '#FCF3CF',         # Soft yellow
    'cluster': '#E8DAEF',    # Soft purple
    'classifier': '#D5F4E6'  # Soft green
}

def draw_rounded_box(x, y, width, height, facecolor, label):
    """Draw a professional rounded box with label"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        facecolor=facecolor,
        edgecolor='#34495E',
        linewidth=3,
        zorder=1
    )
    ax.add_patch(box)

    # Title bar effect
    ax.text(x + width/2, y + height - 0.35, label,
           ha='center', va='top', fontsize=16, fontweight='bold',
           color='#2C3E50')

def draw_thick_arrow(x1, x2, y):
    """Draw thick, professional arrow"""
    arrow = FancyArrowPatch(
        (x1, y), (x2, y),
        arrowstyle='->,head_width=0.25,head_length=0.25',
        color='#34495E',
        linewidth=3,
        zorder=2
    )
    ax.add_patch(arrow)

# Box parameters
y_pos = 0.2
height = 3.5
spacing = 0.35

# ==================== BOX 1: Time Series ====================
x1, w1 = 0, 3.2
draw_rounded_box(x1, y_pos, w1, height, colors['data'], 'Time Series Input')

# Content
ax.text(x1 + w1/2, y_pos + 2.7, r'$\mathbf{X} \in \mathbb{R}^{n \times d}$',
       ha='center', fontsize=14, color='#2C3E50')
ax.text(x1 + w1/2, y_pos + 2.3, r'$n$ samples, $d$ timesteps',
       ha='center', fontsize=12, color='#7F8C8D', style='italic')

# Time series example
ts_x, ts_y = x1 + 0.3, y_pos + 0.4
ts_w, ts_h = w1 - 0.6, 1.4
np.random.seed(42)
t = np.linspace(0, 1, 80)
ts = np.sin(2*np.pi*2*t) + 0.5*np.sin(2*np.pi*5*t)
ax.fill_between(ts_x + ts_w * t, ts_y + ts_h/2, ts_y + ts_h/2 + ts * ts_h/3,
                alpha=0.3, color='#3498DB', linewidth=0)
ax.plot(ts_x + ts_w * t, ts_y + ts_h/2 + ts * ts_h/3,
        linewidth=2.5, color='#2874A6')
ax.plot([ts_x, ts_x + ts_w], [ts_y + ts_h/2, ts_y + ts_h/2],
        'k-', linewidth=1.5, alpha=0.5)
ax.plot([ts_x, ts_x], [ts_y, ts_y + ts_h],
        'k-', linewidth=1.5, alpha=0.5)

draw_thick_arrow(x1 + w1, x1 + w1 + spacing, y_pos + height/2)

# ==================== BOX 2: MiniROCKET ====================
x2 = x1 + w1 + spacing
w2 = 3.2
draw_rounded_box(x2, y_pos, w2, height, colors['transform'], 'MiniROCKET Transform')

ax.text(x2 + w2/2, y_pos + 2.5, r"$\mathbf{X}' = \phi(\mathbf{X})$",
       ha='center', fontsize=14, color='#2C3E50', fontweight='bold')
ax.text(x2 + w2/2, y_pos + 1.95, r"$\phi$: convolutional transform",
       ha='center', fontsize=12, color='#7F8C8D', style='italic')

# Visual representation of features
feat_y = y_pos + 0.6
feat_h = 1.0
n_bars = 20
bar_width = (w2 - 0.8) / n_bars
np.random.seed(10)
for i in range(n_bars):
    bar_height = np.random.uniform(0.3, 0.95) * feat_h
    bar_x = x2 + 0.4 + i * bar_width
    ax.add_patch(Rectangle((bar_x, feat_y), bar_width*0.8, bar_height,
                          facecolor='#E67E22', alpha=0.6, edgecolor='none'))

ax.text(x2 + w2/2, y_pos + 0.25, r'$m \approx 10{,}000$ features',
       ha='center', fontsize=11, color='#7F8C8D', style='italic')

draw_thick_arrow(x2 + w2, x2 + w2 + spacing, y_pos + height/2)

# ==================== BOX 3: JM Distance ====================
x3 = x2 + w2 + spacing
w3 = 4.5
draw_rounded_box(x3, y_pos, w3, height, colors['jm'], 'JM Distance Computation')

# Formulas with better spacing
ax.text(x3 + w3/2, y_pos + 2.6,
       r'Bhattacharyya Distance:',
       ha='center', fontsize=12, color='#7F8C8D', style='italic')

ax.text(x3 + w3/2, y_pos + 2.15,
       r'$B_{ij}^{(k)} = \frac{(\mu_i - \mu_j)^2}{8(\sigma_i^2 + \sigma_j^2)} + \frac{1}{2}\ln\left(\frac{\sigma_i^2 + \sigma_j^2}{2\sigma_i\sigma_j}\right)$',
       ha='center', fontsize=10, color='#2C3E50')

ax.text(x3 + w3/2, y_pos + 1.5,
       r'$JM_{ij}^{(k)} = 2(1 - e^{-B_{ij}^{(k)}})$',
       ha='center', fontsize=13, color='#2C3E50', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                edgecolor='#F39C12', linewidth=1.5))

ax.text(x3 + w3/2, y_pos + 0.9,
       r'$\mathbf{JM} \in \mathbb{R}^{m \times p}$, where $p = \binom{c}{2}$',
       ha='center', fontsize=11, color='#2C3E50')

ax.text(x3 + w3/2, y_pos + 0.4,
       r'$\bar{JM}^{(k)} = \frac{1}{p}\sum_{i<j} JM_{ij}^{(k)}$ (avg per feature)',
       ha='center', fontsize=10, color='#7F8C8D', style='italic')

draw_thick_arrow(x3 + w3, x3 + w3 + spacing, y_pos + height/2)

# ==================== BOX 4: K-means ====================
x4 = x3 + w3 + spacing
w4 = 3.8
draw_rounded_box(x4, y_pos, w4, height, colors['cluster'], 'K-means Clustering')

# Cluster visualization
km_x, km_y = x4 + w4/2 - 0.7, y_pos + 1.2
km_size = 1.4
np.random.seed(123)
cluster_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]

for i, (cx, cy) in enumerate(centers):
    n_points = 12
    px = np.random.normal(cx, 0.05, n_points)
    py = np.random.normal(cy, 0.05, n_points)
    ax.scatter(km_x + px*km_size, km_y + py*km_size,
              c=cluster_colors[i], s=20, alpha=0.5, edgecolors='none', zorder=3)
    # Star for selected feature
    ax.scatter(km_x + cx*km_size, km_y + cy*km_size,
              marker='*', c=cluster_colors[i], s=300,
              edgecolors='#2C3E50', linewidth=1, zorder=4)

# Add axes
ax.plot([km_x, km_x + km_size], [km_y, km_y], 'k-', linewidth=1, alpha=0.3)
ax.plot([km_x, km_x], [km_y, km_y + km_size], 'k-', linewidth=1, alpha=0.3)
ax.text(km_x + km_size + 0.1, km_y, 'Dim 1', fontsize=10, color='#7F8C8D')
ax.text(km_x - 0.15, km_y + km_size, 'Dim 2', fontsize=10, color='#7F8C8D', rotation=90, va='bottom')

ax.text(x4 + w4/2, y_pos + 0.5,
       r'$\mathcal{F} = \{\arg\max_{k \in \mathcal{C}_i} \bar{JM}^{(k)}\}$',
       ha='center', fontsize=11, color='#2C3E50')

ax.text(x4 + w4/2, y_pos + 0.1,
       r'★ = selected feature (K total)',
       ha='center', fontsize=10, color='#7F8C8D', style='italic')

draw_thick_arrow(x4 + w4, x4 + w4 + spacing, y_pos + height/2)

# ==================== BOX 5: Classifier ====================
x5 = x4 + w4 + spacing
w5 = 2.8
draw_rounded_box(x5, y_pos, w5, height, colors['classifier'], 'Classification')

ax.text(x5 + w5/2, y_pos + 2.6, r'Ridge Classifier',
       ha='center', fontsize=14, color='#2C3E50', fontweight='bold')

ax.text(x5 + w5/2, y_pos + 1.8,
       r'$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}_{\mathcal{F}}\mathbf{w}\|^2 + \alpha\|\mathbf{w}\|^2$',
       ha='center', fontsize=12, color='#2C3E50')

ax.text(x5 + w5/2, y_pos + 1.0, r'L2 regularization',
       ha='center', fontsize=11, color='#7F8C8D', style='italic')

# Arrow down to output
ax.annotate('', xy=(x5 + w5/2, y_pos + 0.45), xytext=(x5 + w5/2, y_pos + 0.75),
           arrowprops=dict(arrowstyle='->', lw=2, color='#27AE60'))

ax.text(x5 + w5/2, y_pos + 0.2, r'$\hat{\mathbf{y}}$',
       ha='center', fontsize=14, color='#27AE60', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                edgecolor='#27AE60', linewidth=1.5))


plt.tight_layout()

# Save as PNG (high resolution for LaTeX)
output_path_png = os.path.join(paths.FIGURES_DIR, 'scientific_pipeline_diagram.png')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
print(f"✓ PNG saved to: {output_path_png}")

# Save as PDF (vector format - best for LaTeX papers)
output_path_pdf = os.path.join(paths.FIGURES_DIR, 'scientific_pipeline_diagram.pdf')
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
print(f"✓ PDF saved to: {output_path_pdf}")

plt.show()