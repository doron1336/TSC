import os
"""
Creates a professional workflow diagram for the JM feature selection pipeline.
Style inspired by ROCKET and MiniROCKET papers.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import paths  # noqa: F401  # TSC path config

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors (clean, professional palette)
color_input = '#E8F4F8'  # Light blue
color_transform = '#FFE6CC'  # Light orange
color_jm = '#E6F3E6'  # Light green
color_cluster = '#F0E6FF'  # Light purple
color_output = '#FFE6E6'  # Light red

# Box drawing helper
def draw_box(x, y, width, height, text, color, fontsize=11, fontweight='normal'):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor='black',
                         facecolor=color,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center',
           fontsize=fontsize, fontweight=fontweight,
           wrap=True)

# Arrow drawing helper
def draw_arrow(x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->,head_width=0.4,head_length=0.4',
                           color='black',
                           linewidth=2,
                           connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label,
               fontsize=9, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

# Title
ax.text(5, 11.5, 'JM-based Feature Selection Pipeline',
       ha='center', va='top',
       fontsize=16, fontweight='bold')

# Stage 1: Input Data
draw_box(1, 9.5, 1.5, 1, 'Time Series\nData\n(N samples)', color_input, fontsize=10, fontweight='bold')
ax.text(1.75, 9.2, 'X_train', ha='center', fontsize=8, style='italic')

# Stage 2: MiniROCKET Transform
draw_arrow(2.5, 10, 3.5, 10, 'transform')
draw_box(3.5, 9.5, 2, 1, 'MiniROCKET\nTransform', color_transform, fontsize=11, fontweight='bold')
ax.text(4.5, 9.2, '~9996 features', ha='center', fontsize=8, style='italic')

# Stage 3: JM Computation (detailed breakdown)
draw_arrow(5.5, 10, 6.5, 10, '')

# JM main box
draw_box(6.5, 8, 3, 3.5, '', color_jm, fontsize=10)
ax.text(8, 11.2, 'JM Distance Computation', ha='center', fontsize=11, fontweight='bold')

# JM sub-steps
draw_box(6.8, 10.3, 2.5, 0.6, 'For each feature i:', '#FFFFFF', fontsize=9)
draw_box(6.8, 9.5, 2.5, 0.6, '1. Calculate JM matrix\n(all class pairs)', '#FFFFFF', fontsize=9)
draw_box(6.8, 8.7, 2.5, 0.6, '2. Extract upper triangle', '#FFFFFF', fontsize=9)
draw_box(6.8, 7.9, 2.5, 0.6, '3. Flatten to vector', '#FFFFFF', fontsize=9)

# JM formula box (side annotation)
formula_text = (
    'JM Distance:\n'
    r'$JM = 2(1 - e^{-B})$'
)
ax.text(9.7, 9.5, formula_text, ha='left', fontsize=9,
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFFCC', edgecolor='black', linewidth=1))

# Stage 4: JM Flat Data
draw_arrow(8, 8, 8, 6.8, '')
draw_box(6.5, 5.8, 3, 0.8, 'JM Flat Data\n(9996 × n_pairs)', color_jm, fontsize=10, fontweight='bold')
ax.text(8, 5.4, 'n_pairs = C(n_classes, 2)', ha='center', fontsize=8, style='italic')

# Stage 5: K-means Clustering
draw_arrow(8, 5.8, 8, 4.5, 'standardize')
draw_box(6.5, 3, 3, 1.3, 'K-means Clustering\n(K clusters)', color_cluster, fontsize=11, fontweight='bold')
ax.text(8, 2.7, 'Group features by\nsimilar separability profiles', ha='center', fontsize=8, style='italic')

# Stage 6: Feature Selection
draw_arrow(8, 3, 8, 2, '')
draw_box(6.5, 1, 3, 0.8, 'Select Best from\nEach Cluster', color_cluster, fontsize=10, fontweight='bold')
ax.text(8, 0.6, 'Highest avg JM per cluster', ha='center', fontsize=8, style='italic')

# Stage 7: Final Output
draw_arrow(8, 1, 5, 0.3, '')
draw_box(3, 0, 4, 0.5, 'K Selected Features (diverse & discriminative)', color_output, fontsize=10, fontweight='bold')

# Add example annotation on the left
example_box = (
    'Example:\n'
    '• 6 classes → 15 pairs\n'
    '• JM matrix: 6×6\n'
    '• Flat vector: 15 values\n'
    '• One vector per feature'
)
ax.text(0.5, 6.5, example_box, ha='left', va='top', fontsize=8,
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='gray', linewidth=1))

# Add key insight box on the right (bottom)
insight_box = (
    'Key Insight:\n'
    'JM measures class separability.\n'
    'K-means ensures diversity.\n'
    'Result: Non-redundant,\n'
    'discriminative features.'
)
ax.text(0.5, 3, insight_box, ha='left', va='top', fontsize=8,
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6', edgecolor='orange', linewidth=1.5))

plt.tight_layout()
plt.savefig(os.path.join(paths.FIGURES_DIR, 'jm_workflow_diagram.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("Workflow diagram saved to: visualizations/jm_workflow_diagram.png")
plt.show()