import os
import numpy as np
from AlgorithmManager import AlgorithmManager

def algo_results_manager(dataset: str, target_subfolder: str, file_name: str):
    return AlgorithmManager.load(os.path.join(dataset, target_subfolder, file_name))

# Configuration
baseDir = "UCRArchive_2018/HAR_datasets"
dataset_name = 'CricketY'
NUM_FOLDS = 5
FEATURE_INDEX = 10  # Index 10 = 210 features in range(10, 211, 20)

# Algorithms to check
algorithms = ['kmeans_avg_jm', 'mrmr', 'relieff']

# Change to base directory
original_dir = os.getcwd()
os.chdir(os.path.join(original_dir, baseDir))

print(f"Duration Analysis for {dataset_name} at 210 features")
print("="*80)
print()

for algorithm_name in algorithms:
    print(f"\n{'='*80}")
    print(f"Algorithm: {algorithm_name}")
    print(f"{'='*80}")

    all_durations = []
    duration_details = []

    # Collect durations across all folds
    for fold_num in range(1, NUM_FOLDS + 1):
        try:
            results = algo_results_manager(dataset=dataset_name, target_subfolder=str(fold_num),
                                           file_name='algo_manager')
            durations = results.get_durations(algorithm_name)

            if durations and len(durations) > FEATURE_INDEX:
                duration = durations[FEATURE_INDEX]
                all_durations.append(duration)
                duration_details.append((fold_num, duration))
            else:
                print(f"  Warning: Fold {fold_num} - insufficient duration data")

        except Exception as e:
            print(f"  Error loading fold {fold_num}: {e}")

    if len(all_durations) < 2:
        print(f"  Insufficient data for {algorithm_name}")
        continue

    # Calculate statistics with all folds
    mean_all = np.mean(all_durations)
    std_all = np.std(all_durations, ddof=1)
    var_all = np.var(all_durations, ddof=1)

    print(f"\n1. WITH ALL FOLDS ({len(all_durations)} folds):")
    print(f"   {'-'*70}")
    for fold_num, duration in duration_details:
        outlier_marker = ""
        if len(all_durations) > 2:
            # Mark potential outliers (> 3 std from mean)
            if abs(duration - mean_all) > 3 * std_all:
                outlier_marker = " ⚠️ OUTLIER"
        print(f"   Fold {fold_num}: {duration:.10f} seconds{outlier_marker}")

    print(f"\n   Statistics:")
    print(f"   Mean:               {mean_all:.10f} seconds")
    print(f"   Standard Deviation: {std_all:.10f} seconds")
    print(f"   Variance:           {var_all:.10f} seconds²")
    print(f"   Min:                {np.min(all_durations):.10f} seconds")
    print(f"   Max:                {np.max(all_durations):.10f} seconds")
    print(f"\n   LaTeX: ${{mean_all:.4f}} \\pm {{std_all:.4f}}$")
    print(f"   LaTeX: ${mean_all:.4f} \\pm {std_all:.4f}$")

    # Detect and remove outliers (values > 3 std from mean)
    if len(all_durations) > 2:
        threshold = 3 * std_all
        cleaned_durations = [d for d in all_durations if abs(d - mean_all) < threshold]

        if len(cleaned_durations) < len(all_durations):
            print(f"\n2. WITHOUT OUTLIERS ({len(cleaned_durations)} folds):")
            print(f"   {'-'*70}")

            mean_clean = np.mean(cleaned_durations)
            std_clean = np.std(cleaned_durations, ddof=1) if len(cleaned_durations) > 1 else 0
            var_clean = np.var(cleaned_durations, ddof=1) if len(cleaned_durations) > 1 else 0

            print(f"   Removed {len(all_durations) - len(cleaned_durations)} outlier(s)")
            print(f"\n   Statistics:")
            print(f"   Mean:               {mean_clean:.10f} seconds")
            print(f"   Standard Deviation: {std_clean:.10f} seconds")
            print(f"   Variance:           {var_clean:.10f} seconds²")
            print(f"   Min:                {np.min(cleaned_durations):.10f} seconds")
            print(f"   Max:                {np.max(cleaned_durations):.10f} seconds")
            print(f"\n   LaTeX: ${mean_clean:.4f} \\pm {std_clean:.4f}$")
            print(f"   LaTeX: ${mean_clean:.6f} \\pm {std_clean:.6f}$")

            print(f"\n   IMPROVEMENT:")
            print(f"   Mean reduction:     {mean_all - mean_clean:.4f} seconds ({(mean_all - mean_clean)/mean_all*100:.1f}%)")
            print(f"   Std reduction:      {std_all - std_clean:.4f} seconds ({(std_all - std_clean)/std_all*100:.1f}%)")

print(f"\n{'='*80}")
print("Analysis complete!")
print(f"{'='*80}")

# Change back to original directory
os.chdir(original_dir)