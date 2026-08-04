"""
Simple, focused visualization showing why JM is beneficial.
Shows a single heatmap comparing high JM vs low JM features.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils.JM import JM_matrix, JM_flat
from utils.retrieve_minirocket import retrieve_minirocket_data

# Configuration
DATASET_NAME = 'UWaveGestureLibraryAll'  # Can be changed to any HAR dataset
DIRECTORY_NUM = 3
BASE_DIR = '/Users/doron/Desktop/personal/thesis/TSC/UCRArchive_2018'

print(f"Loading dataset: {DATASET_NAME}")
directory = os.path.join(BASE_DIR, DATASET_NAME, str(DIRECTORY_NUM))
X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(directory, DATASET_NAME)

# Normalize labels to start from 0 and convert to strings (required by JM_matrix function)
# This matches what JM_flat does internally
gt_new = [int(i) for i in y_train]
if np.min(gt_new) != 0:
    y_train = np.array([str(int(i)-np.min(gt_new)) for i in gt_new])
else:
    y_train = np.array([str(int(i)) for i in y_train])

gt_test = [int(i) for i in y_test]
if np.min(gt_test) != 0:
    y_test = np.array([str(int(i)-np.min(gt_test)) for i in gt_test])
else:
    y_test = np.array([str(int(i)) for i in y_test])

print(f"Label types: {type(y_train[0])}, unique labels: {np.unique(y_train)}")
print(f"Label distribution: {np.bincount([int(y) for y in y_train])}")

# Load JM data
avg_jm_cache = os.path.join(directory, 'MMD', 'avg_jm')
if os.path.exists(avg_jm_cache):
    print("Loading cached JM data...")
    with open(avg_jm_cache, 'rb') as file:
        avg_jm_per_feature = pickle.load(file)
else:
    print("Computing JM flat data (this may take a while)...")
    JM_flat_data, avg_jm_per_feature = JM_flat(X_train_transform, y_train)
    print("✓ JM computation complete")

# Get dataset info
classes = np.unique(y_train)
n_classes = len(classes)
n_features = X_train_transform.shape[1]

print(f"Dataset: {n_classes} classes, {n_features} features")

# Select best and worst features
best_feature_idx = np.argmax(avg_jm_per_feature)
worst_feature_idx = np.argmin(avg_jm_per_feature)

# Compute JM matrices
print(f"Computing JM matrices...")
print(f"  y_train sample: {y_train[:10]}")
print(f"  Feature data shape: {X_train_transform[:, best_feature_idx].shape}")
print(f"  Feature data sample: {X_train_transform[:10, best_feature_idx]}")

best_jm_matrix = JM_matrix(X_train_transform[:, best_feature_idx], y_train, best_feature_idx)
worst_jm_matrix = JM_matrix(X_train_transform[:, worst_feature_idx], y_train, worst_feature_idx)

print(f"  Best JM matrix:\n{best_jm_matrix}")
print(f"  Matrix sum: {np.sum(best_jm_matrix)}, non-zero elements: {np.count_nonzero(best_jm_matrix)}")

# Get average JM scores
best_avg_jm = avg_jm_per_feature[best_feature_idx]
worst_avg_jm = avg_jm_per_feature[worst_feature_idx]

print(f"Best feature: #{best_feature_idx}, avg JM={best_avg_jm:.3f}")
print(f"Worst feature: #{worst_feature_idx}, avg JM={worst_avg_jm:.3f}")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color map settings
vmin, vmax = 0, 2
cmap = 'RdYlGn'

# Plot 1: High JM feature
ax1 = axes[0]
im1 = ax1.imshow(best_jm_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

# Add text annotations
for i in range(n_classes):
    for j in range(n_classes):
        if i != j:
            value = best_jm_matrix[i, j]
            text_color = 'white' if value < 1.0 else 'black'
            ax1.text(j, i, f'{value:.2f}',
                     ha="center", va="center",
                     color=text_color, fontsize=11, fontweight='bold')
        else:
            ax1.text(j, i, '—',
                     ha="center", va="center",
                     color='gray', fontsize=11)

ax1.set_title(f'High Discriminative Feature (Feature #{best_feature_idx})\n'
              f'Average JM = {best_avg_jm:.3f}',
              fontsize=13, fontweight='bold', color='darkgreen', pad=15)
ax1.set_xlabel('Class', fontsize=11, fontweight='bold')
ax1.set_ylabel('Class', fontsize=11, fontweight='bold')
ax1.set_xticks(range(n_classes))
ax1.set_yticks(range(n_classes))
ax1.set_xticklabels(range(n_classes), fontsize=10)
ax1.set_yticklabels(range(n_classes), fontsize=10)

# Add colorbar
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('JM Distance', fontsize=10, fontweight='bold')
cbar1.ax.tick_params(labelsize=9)

# Add interpretation box
ax1.text(0.5, -0.15,
         'High JM values (green) indicate strong class separation\n→ Feature captures discriminative patterns',
         transform=ax1.transAxes, ha='center', va='top', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='darkgreen', linewidth=2))

# Plot 2: Low JM feature
ax2 = axes[1]
im2 = ax2.imshow(worst_jm_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

# Add text annotations
for i in range(n_classes):
    for j in range(n_classes):
        if i != j:
            value = worst_jm_matrix[i, j]
            text_color = 'white' if value < 1.0 else 'black'
            ax2.text(j, i, f'{value:.2f}',
                     ha="center", va="center",
                     color=text_color, fontsize=11, fontweight='bold')
        else:
            ax2.text(j, i, '—',
                     ha="center", va="center",
                     color='gray', fontsize=11)

ax2.set_title(f'Low Discriminative Feature (Feature #{worst_feature_idx})\n'
              f'Average JM = {worst_avg_jm:.3f}',
              fontsize=13, fontweight='bold', color='darkred', pad=15)
ax2.set_xlabel('Class', fontsize=11, fontweight='bold')
ax2.set_ylabel('Class', fontsize=11, fontweight='bold')
ax2.set_xticks(range(n_classes))
ax2.set_yticks(range(n_classes))
ax2.set_xticklabels(range(n_classes), fontsize=10)
ax2.set_yticklabels(range(n_classes), fontsize=10)

# Add colorbar
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('JM Distance', fontsize=10, fontweight='bold')
cbar2.ax.tick_params(labelsize=9)

# Add interpretation box
ax2.text(0.5, -0.15,
         'Low JM values (red/yellow) indicate poor class separation\n→ Feature provides little discriminative information',
         transform=ax2.transAxes, ha='center', va='top', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', edgecolor='darkred', linewidth=2))

# Main title
fig.suptitle(f'JM Distance Matrix: Quantifying Feature Discriminative Power\n'
             f'Dataset: {DATASET_NAME}',
             fontsize=15, fontweight='bold', y=0.98)

# Add explanation at the bottom
# explanation = (
#     'JM Distance Interpretation:\n'
#     '• JM ∈ [0, 2]: 0 = identical class distributions, 2 = complete separation\n'
#     '• Symmetric matrix: JM(i,j) = JM(j,i)\n'
#     '• Diagonal = 0: each class is identical to itself\n'
#     '• High avg JM → feature separates classes well → useful for classification'
# )
# fig.text(0.5, 0.02, explanation, ha='center', va='bottom', fontsize=10, family='monospace',
#          bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', edgecolor='black', linewidth=1.5))

plt.tight_layout(rect=[0, 0.12, 1, 0.95])

# Save figure
output_path = f'/Users/doron/Desktop/personal/thesis/TSC/visualizations/jm_heatmap_comparison_{DATASET_NAME}_1.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ JM heatmap saved to: {output_path}")

plt.show()

# Print summary
print("\n" + "=" * 60)
print("JM HEATMAP COMPARISON SUMMARY")
print("=" * 60)
print(f"Dataset: {DATASET_NAME} ({n_classes} classes)")
print(f"\nHigh Discriminative Feature:")
print(f"  Feature #{best_feature_idx}")
print(f"  Average JM: {best_avg_jm:.4f}")
print(f"  Max pairwise JM: {np.max(best_jm_matrix[np.triu_indices_from(best_jm_matrix, k=1)]):.4f}")
print(f"  Min pairwise JM: {np.min(best_jm_matrix[np.triu_indices_from(best_jm_matrix, k=1)]):.4f}")
print(f"\nLow Discriminative Feature:")
print(f"  Feature #{worst_feature_idx}")
print(f"  Average JM: {worst_avg_jm:.4f}")
print(f"  Max pairwise JM: {np.max(worst_jm_matrix[np.triu_indices_from(worst_jm_matrix, k=1)]):.4f}")
print(f"  Min pairwise JM: {np.min(worst_jm_matrix[np.triu_indices_from(worst_jm_matrix, k=1)]):.4f}")
print(f"\nDifference in avg JM: {best_avg_jm - worst_avg_jm:.4f}")
print("=" * 60)
