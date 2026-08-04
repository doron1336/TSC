import os
import pandas as pd
from AlgorithmManager import AlgorithmManager

# Test the aggregate_scores logic with actual data
path = '/Users/doron/Desktop/personal/thesis/TSC/UCRArchive_2018'
os.chdir(path)
HAR_directory = os.path.join(os.getcwd(), "HAR_datasets")

dataset_name = "GestureMidAirD3"
algo_name = "kmeans_avg_jm"
dataset_directory = os.path.join(HAR_directory, dataset_name)

print(f"Debugging {algo_name} for {dataset_name}\n")

scores_dict = {}
for i in range(1, 6):
    results = AlgorithmManager.load(os.path.join(dataset_directory, str(i), 'algo_manager'))
    predictions = results.get_predictions(algo_name)
    print(f"Fold {i}: {len(predictions)} predictions")
    print(f"  Values: {predictions}\n")
    scores_dict[i] = predictions

# Create DataFrame the same way as results_table.py
df = pd.DataFrame.from_dict(scores_dict).T
print("DataFrame structure:")
print(df)
print(f"\nDataFrame shape: {df.shape}")
print(f"Rows (folds): {len(df)}, Columns (feature counts): {len(df.columns)}\n")

# Compute mean and std
means = df.mean()
stds = df.std()

print("Means across folds:")
print(means)
print("\nStd across folds:")
print(stds)

# Show formatted output like in the table
print("\nFormatted output (as in table):")
for col in df.columns:
    print(f"Feature count {col}: {means[col]:.4f} ± {stds[col]:.4f}")