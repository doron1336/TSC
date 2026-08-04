"""
Comprehensive visualization showing how JM helps understand feature quality
and class separability. This creates a publication-ready figure demonstrating:
1. Feature distributions with different JM scores
2. JM matrices showing pairwise class separability
3. Feature ranking and selection
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from TSC.utils.JM import JM_flat, JM_matrix, computeJM
from TSC.utils.retrieve_minirocket import retrieve_minirocket_data

# Configuration - change these as needed
DATASET_NAME = 'GestureMidAirD3'
DIRECTORY_NUM = 3
BASE_DIR = str(paths.UCR_DIR)
print(f"Loading dataset: {DATASET_NAME}")
directory = os.path.join(BASE_DIR, DATASET_NAME, str(DIRECTORY_NUM))

# Load MiniROCKET transformed data
X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(directory, DATASET_NAME)

# Load or compute JM data
jm_cache_path = os.path.join(directory, 'MMD', 'JM_flat_data_full')
avg_jm_cache = os.path.join(directory, 'MMD', 'avg_jm')

if os.path.exists(jm_cache_path) and os.path.exists(avg_jm_cache):
    print("Loading cached JM data...")
    with open(jm_cache_path, 'rb') as file:
        JM_flat_data = pickle.load(file)
    with open(avg_jm_cache, 'rb') as file:
        avg_jm_per_feature = pickle.load(file)
else:
    print("Computing JM flat data (this may take a while)...")
    JM_flat_data, avg_jm_per_feature = JM_flat(X_train_transform, y_train)
    print("✓ JM computation complete")

# Dataset info
n_features = X_train_transform.shape[1]
n_samples = X_train_transform.shape[0]
classes = np.unique(y_train)
n_classes = len(classes)
n_pairs = int(n_classes * (n_classes - 1) / 2)

print(f"\nDataset: {n_samples} samples, {n_features} features, {n_classes} classes")
print(f"JM statistics - Mean: {np.mean(avg_jm_per_feature):.3f}, "
      f"Median: {np.median(avg_jm_per_feature):.3f}, "
      f"Max: {np.max(avg_jm_per_feature):.3f}")

# Select representative features
top_features_indices = np.argsort(avg_jm_per_feature)[-10:]  # Top 10
best_feature = top_features_indices[-1]
good_feature = top_features_indices[-3]
median_idx = len(avg_jm_per_feature) // 2
mediocre_feature = np.argsort(avg_jm_per_feature)[median_idx]
poor_feature = np.argsort(avg_jm_per_feature)[100]  # 100th worst

selected_features = [
    (best_feature, 'High JM (Best)', 'darkgreen'),
    (good_feature, 'High JM (3rd)', 'green'),
    (mediocre_feature, 'Medium JM', 'orange'),
    (poor_feature, 'Low JM', 'red')
]

# Create figure
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4,
             left=0.06, right=0.96, top=0.94, bottom=0.06)

# Normalize class labels
y_train_normalized = np.array([int(y) - int(classes[0]) for y in y_train])
class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

# ============= Column 1: Feature Distributions =============
for idx, (feature_idx, title, title_color) in enumerate(selected_features):
    ax = fig.add_subplot(gs[idx, 0])

    avg_jm = avg_jm_per_feature[feature_idx]

    # Plot overlapping histograms
    for class_idx in range(n_classes):
        class_mask = y_train_normalized == class_idx
        feature_values = X_train_transform[class_mask, feature_idx]
        ax.hist(feature_values, bins=25, alpha=0.5, label=f'C{class_idx}',
               color=class_colors[class_idx], edgecolor='black', linewidth=0.3)

    ax.set_title(f'{title}\nFeature {feature_idx} (JM={avg_jm:.3f})',
                fontsize=10, fontweight='bold', color=title_color)
    ax.set_xlabel('Feature Value', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    if idx == 0:
        ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ============= Column 2: JM Matrices =============
for idx, (feature_idx, title, title_color) in enumerate(selected_features):
    ax = fig.add_subplot(gs[idx, 1])

    # Compute JM matrix
    feature_data = X_train_transform[:, feature_idx]
    jm_matrix = JM_matrix(feature_data, y_train, feature_idx)

    # Plot heatmap
    im = ax.imshow(jm_matrix, cmap='RdYlGn', vmin=0, vmax=2, aspect='auto')

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                value = jm_matrix[i, j]
                text_color = 'white' if value < 1.0 else 'black'
                ax.text(j, i, f'{value:.2f}',
                       ha="center", va="center",
                       color=text_color, fontsize=8, fontweight='bold')

    ax.set_title(f'JM Matrix\n(avg={np.mean(jm_matrix[np.triu_indices_from(jm_matrix, k=1)]):.3f})',
                fontsize=9, fontweight='bold')
    ax.set_xlabel('Class', fontsize=9)
    ax.set_ylabel('Class', fontsize=9)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))

    # Colorbar
    if idx == 0:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('JM Distance', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

# ============= Column 3: Analysis Plots =============

# Plot 1: Feature ranking
ax1 = fig.add_subplot(gs[0, 2:])
sorted_indices = np.argsort(avg_jm_per_feature)[::-1]
top_n = 100
x_range = range(top_n)
bars = ax1.bar(x_range, avg_jm_per_feature[sorted_indices[:top_n]],
              color='steelblue', edgecolor='black', linewidth=0.3, alpha=0.7)

# Highlight selected features in the ranking
for feature_idx, title, color in selected_features:
    rank = np.where(sorted_indices == feature_idx)[0][0]
    if rank < top_n:
        bars[rank].set_color(color)
        bars[rank].set_linewidth(1.5)

ax1.set_title('Top 100 Features Ranked by Average JM Distance',
             fontsize=11, fontweight='bold')
ax1.set_xlabel('Feature Rank', fontsize=10)
ax1.set_ylabel('Average JM', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add legend for highlighted features
from matplotlib.patches import Patch
import paths  # noqa: F401  # TSC path config (TSC_CODE_DIR, TSC_DATA_DIR, TSC_RESULTS_DIR)
legend_elements = [Patch(facecolor=color, edgecolor='black', label=title)
                  for _, title, color in selected_features]
ax1.legend(handles=legend_elements, fontsize=8, loc='upper right')

# Plot 2: Distribution of JM scores
ax2 = fig.add_subplot(gs[1, 2:])
hist_vals, bins, patches = ax2.hist(avg_jm_per_feature, bins=60,
                                     color='coral', edgecolor='black',
                                     linewidth=0.3, alpha=0.7)

# Color the histogram bins
for i, (patch, val) in enumerate(zip(patches, hist_vals)):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center > 1.5:
        patch.set_facecolor('darkgreen')
    elif bin_center > 1.0:
        patch.set_facecolor('green')
    elif bin_center > 0.5:
        patch.set_facecolor('orange')
    else:
        patch.set_facecolor('red')

mean_val = np.mean(avg_jm_per_feature)
median_val = np.median(avg_jm_per_feature)
ax2.axvline(mean_val, color='blue', linestyle='--', linewidth=2,
           label=f'Mean={mean_val:.3f}')
ax2.axvline(median_val, color='purple', linestyle='--', linewidth=2,
           label=f'Median={median_val:.3f}')

ax2.set_title('Distribution of Average JM Across All Features',
             fontsize=11, fontweight='bold')
ax2.set_xlabel('Average JM Distance', fontsize=10)
ax2.set_ylabel('Number of Features', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Plot 3: Class-pair separability for best feature
ax3 = fig.add_subplot(gs[2, 2:])
best_feature_jm = JM_flat_data[best_feature]

# Create class pair labels
pair_labels = []
for i in range(n_classes):
    for j in range(i+1, n_classes):
        pair_labels.append(f'{i}-{j}')

x_pos = np.arange(len(pair_labels))
bars = ax3.bar(x_pos, best_feature_jm, color='mediumseagreen',
              edgecolor='black', linewidth=0.5, alpha=0.8)

# Color bars based on JM value
for i, (bar, jm_val) in enumerate(zip(bars, best_feature_jm)):
    if jm_val > 1.5:
        bar.set_color('darkgreen')
    elif jm_val > 1.0:
        bar.set_color('green')
    elif jm_val > 0.5:
        bar.set_color('orange')
    else:
        bar.set_color('red')

ax3.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5,
           label='JM=1.0 (moderate separation)', zorder=0)
ax3.axhline(y=1.5, color='black', linestyle=':', linewidth=1.5,
           label='JM=1.5 (good separation)', zorder=0)

ax3.set_title(f'Class-Pair Separability for Best Feature (#{best_feature})\n'
             f'Overall avg JM = {avg_jm_per_feature[best_feature]:.3f}',
             fontsize=11, fontweight='bold')
ax3.set_xlabel('Class Pair', fontsize=10)
ax3.set_ylabel('JM Distance', fontsize=10)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(pair_labels, rotation=45, fontsize=8)
ax3.grid(True, alpha=0.3, axis='y', linewidth=0.5)
ax3.legend(fontsize=8, loc='lower right')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Plot 4: Comparison of selected features' separability profiles
ax4 = fig.add_subplot(gs[3, 2:])

for feature_idx, title, color in selected_features:
    jm_profile = JM_flat_data[feature_idx]
    ax4.plot(x_pos, jm_profile, marker='o', linewidth=2,
            markersize=5, label=f'{title} (avg={avg_jm_per_feature[feature_idx]:.2f})',
            color=color, alpha=0.7)

ax4.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax4.set_title('JM Separability Profiles Comparison\n(Different features show different class-pair separability patterns)',
             fontsize=11, fontweight='bold')
ax4.set_xlabel('Class Pair', fontsize=10)
ax4.set_ylabel('JM Distance', fontsize=10)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(pair_labels, rotation=45, fontsize=8)
ax4.legend(fontsize=8, loc='best')
ax4.grid(True, alpha=0.3, linewidth=0.5)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Main title
fig.suptitle(f'JM-based Feature Quality Analysis: Understanding Class Separability\n'
            f'Dataset: {DATASET_NAME} ({n_samples} samples, {n_classes} classes, {n_features} features)',
            fontsize=14, fontweight='bold')

# Save figure
output_path = os.path.join(paths.FIGURES_DIR, f'jm_separability_analysis_{DATASET_NAME}.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ JM separability visualization saved to: {output_path}")

plt.show()

# Print detailed summary
print("\n" + "="*70)
print("JM FEATURE ANALYSIS SUMMARY")
print("="*70)
print(f"\nDataset: {DATASET_NAME}")
print(f"  Samples: {n_samples} | Features: {n_features} | Classes: {n_classes}")
print(f"\nJM Statistics across all {n_features} features:")
print(f"  Mean:   {np.mean(avg_jm_per_feature):.4f}")
print(f"  Median: {np.median(avg_jm_per_feature):.4f}")
print(f"  Std:    {np.std(avg_jm_per_feature):.4f}")
print(f"  Min:    {np.min(avg_jm_per_feature):.4f}")
print(f"  Max:    {np.max(avg_jm_per_feature):.4f}")

print(f"\nSelected features for visualization:")
for feature_idx, title, color in selected_features:
    avg_jm = avg_jm_per_feature[feature_idx]
    rank = np.where(np.argsort(avg_jm_per_feature)[::-1] == feature_idx)[0][0] + 1
    print(f"  Feature {feature_idx:5d} | Rank: {rank:5d}/{n_features} | "
          f"Avg JM: {avg_jm:.4f} | {title}")

print(f"\nBest feature (#{best_feature}):")
best_jm_profile = JM_flat_data[best_feature]
max_idx = np.argmax(best_jm_profile)
min_idx = np.argmin(best_jm_profile)
pair_labels_list = [f'{i}-{j}' for i in range(n_classes) for j in range(i+1, n_classes)]
print(f"  Most separable pair:  {pair_labels_list[max_idx]} (JM={best_jm_profile[max_idx]:.4f})")
print(f"  Least separable pair: {pair_labels_list[min_idx]} (JM={best_jm_profile[min_idx]:.4f})")
print("="*70)