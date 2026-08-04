"""
Demonstrates how JM distance helps understand which features separate between classes.
Creates multiple subplots showing:
1. Feature distributions for different classes
2. JM matrix heatmap
3. Feature ranking by average JM
4. Class-pair specific separability
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from utils.JM import JM_flat, JM_matrix, computeJM
from utils.retrieve_minirocket import retrieve_minirocket_data

# Configuration
DATASET_NAME = 'UWaveGestureLibraryAll'  # Can be changed to any HAR dataset
DIRECTORY_NUM = 3
BASE_DIR = '/Users/doron/Desktop/personal/thesis/TSC/UCRArchive_2018'

# Load data
print(f"Loading dataset: {DATASET_NAME}")
directory = os.path.join(BASE_DIR, DATASET_NAME, str(DIRECTORY_NUM))
X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(directory, DATASET_NAME)

# Compute JM flat data if not cached
jm_cache_path = os.path.join(directory, 'MMD', 'JM_flat_data_full')
if os.path.exists(jm_cache_path):
    print("Loading cached JM data...")
    with open(jm_cache_path, 'rb') as file:
        JM_flat_data = pickle.load(file)
else:
    print("Computing JM flat data (this may take a while)...")
    JM_flat_data, _ = JM_flat(X_train_transform, y_train)

# Get basic info
n_features = X_train_transform.shape[1]
n_samples = X_train_transform.shape[0]
classes = np.unique(y_train)
n_classes = len(classes)
n_pairs = int(n_classes * (n_classes - 1) / 2)

print(f"Dataset info: {n_samples} samples, {n_features} features, {n_classes} classes")

# Calculate average JM per feature
avg_jm_per_feature = np.mean(JM_flat_data, axis=1)

# Select interesting features to visualize
top_features = np.argsort(avg_jm_per_feature)[-3:]  # Top 3 features
random_feature = np.random.choice(np.argsort(avg_jm_per_feature)[len(avg_jm_per_feature)//2:len(avg_jm_per_feature)//2+100])  # A mediocre feature

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

# ============= Row 1: Feature distributions for different quality features =============

# Normalize class labels to start from 0
y_train_normalized = np.array([int(y) - int(classes[0]) for y in y_train])

for idx, (feature_idx, title_suffix) in enumerate([(top_features[0], 'Best Feature'),
                                                     (random_feature, 'Mediocre Feature'),
                                                     (top_features[-1], '3rd Best Feature')]):
    ax = fig.add_subplot(gs[0, idx])

    # Plot distribution for each class
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for class_idx in range(n_classes):
        class_mask = y_train_normalized == class_idx
        feature_values = X_train_transform[class_mask, feature_idx]
        ax.hist(feature_values, bins=30, alpha=0.6, label=f'Class {class_idx}',
               color=colors[class_idx], edgecolor='black', linewidth=0.5)

    avg_jm = avg_jm_per_feature[feature_idx]
    ax.set_title(f'{title_suffix}\nFeature {feature_idx} (avg JM={avg_jm:.3f})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Feature Value', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

# ============= Row 2: JM Matrices for the selected features =============

for idx, (feature_idx, title_suffix) in enumerate([(top_features[0], 'Best'),
                                                     (random_feature, 'Mediocre'),
                                                     (top_features[-1], '3rd Best')]):
    ax = fig.add_subplot(gs[1, idx])

    # Compute JM matrix for this feature
    feature_data = X_train_transform[:, feature_idx]
    jm_matrix = JM_matrix(feature_data, y_train, feature_idx)

    # Plot heatmap
    im = ax.imshow(jm_matrix, cmap='RdYlGn', vmin=0, vmax=2, aspect='auto')

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                text = ax.text(j, i, f'{jm_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=7)

    ax.set_title(f'{title_suffix} Feature\nJM Matrix', fontsize=10, fontweight='bold')
    ax.set_xlabel('Class', fontsize=9)
    ax.set_ylabel('Class', fontsize=9)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('JM Distance', fontsize=8)

# ============= Row 3: Analysis plots =============

# Plot 1: Feature ranking by average JM
ax1 = fig.add_subplot(gs[2, 0])
sorted_indices = np.argsort(avg_jm_per_feature)[::-1]
top_n = 50
x_range = range(top_n)
ax1.bar(x_range, avg_jm_per_feature[sorted_indices[:top_n]],
       color='steelblue', edgecolor='black', linewidth=0.5)
ax1.set_title('Top 50 Features by Average JM', fontsize=10, fontweight='bold')
ax1.set_xlabel('Feature Rank', fontsize=9)
ax1.set_ylabel('Average JM Distance', fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Distribution of average JM across all features
ax2 = fig.add_subplot(gs[2, 1])
ax2.hist(avg_jm_per_feature, bins=50, color='coral', edgecolor='black', linewidth=0.5)
ax2.axvline(np.mean(avg_jm_per_feature), color='red', linestyle='--',
           linewidth=2, label=f'Mean={np.mean(avg_jm_per_feature):.3f}')
ax2.axvline(np.median(avg_jm_per_feature), color='blue', linestyle='--',
           linewidth=2, label=f'Median={np.median(avg_jm_per_feature):.3f}')
ax2.set_title('Distribution of Avg JM\nAcross All Features', fontsize=10, fontweight='bold')
ax2.set_xlabel('Average JM Distance', fontsize=9)
ax2.set_ylabel('Number of Features', fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Class-pair separability for best feature
ax3 = fig.add_subplot(gs[2, 2])
best_feature_idx = top_features[0]
best_feature_jm = JM_flat_data[best_feature_idx]

# Create class pair labels
pair_labels = []
for i in range(n_classes):
    for j in range(i+1, n_classes):
        pair_labels.append(f'{i}-{j}')

x_pos = np.arange(len(pair_labels))
bars = ax3.bar(x_pos, best_feature_jm, color='mediumseagreen',
              edgecolor='black', linewidth=0.5)

# Highlight the best and worst separable pairs
max_idx = np.argmax(best_feature_jm)
min_idx = np.argmin(best_feature_jm)
bars[max_idx].set_color('darkgreen')
bars[min_idx].set_color('darkred')

ax3.set_title(f'Class-Pair Separability\nBest Feature ({best_feature_idx})',
             fontsize=10, fontweight='bold')
ax3.set_xlabel('Class Pair', fontsize=9)
ax3.set_ylabel('JM Distance', fontsize=9)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(pair_labels, rotation=45, fontsize=7)
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='JM=1.0')
ax3.legend(fontsize=7)

# Add main title
fig.suptitle(f'JM-based Feature Separability Analysis\nDataset: {DATASET_NAME}',
            fontsize=14, fontweight='bold', y=0.98)

# Save figure
output_path = f'/Users/doron/Desktop/personal/thesis/TSC/visualizations/jm_class_separation_{DATASET_NAME}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nClass separation visualization saved to: {output_path}")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total features: {n_features}")
print(f"Number of classes: {n_classes}")
print(f"Number of class pairs: {n_pairs}")
print(f"\nAverage JM statistics:")
print(f"  Mean: {np.mean(avg_jm_per_feature):.4f}")
print(f"  Median: {np.median(avg_jm_per_feature):.4f}")
print(f"  Std: {np.std(avg_jm_per_feature):.4f}")
print(f"  Min: {np.min(avg_jm_per_feature):.4f}")
print(f"  Max: {np.max(avg_jm_per_feature):.4f}")
print(f"\nBest feature: {best_feature_idx} (avg JM={avg_jm_per_feature[best_feature_idx]:.4f})")
print(f"  Most separable pair: {pair_labels[max_idx]} (JM={best_feature_jm[max_idx]:.4f})")
print(f"  Least separable pair: {pair_labels[min_idx]} (JM={best_feature_jm[min_idx]:.4f})")
print("="*60)