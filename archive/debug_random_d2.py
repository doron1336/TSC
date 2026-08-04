import os
import pandas as pd
from AlgorithmManager import AlgorithmManager

# Check random algo for GestureMidAirD2
path = '/Users/doron/Desktop/personal/thesis/TSC/UCRArchive_2018'
os.chdir(path)
HAR_directory = os.path.join(os.getcwd(), "HAR_datasets")

dataset_name = "GestureMidAirD2"
algo_name = "random"
dataset_directory = os.path.join(HAR_directory, dataset_name)

print(f"Checking {algo_name} for {dataset_name}\n")
print("Feature counts: 10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210\n")

scores_dict = {}
for i in range(1, 6):
    results = AlgorithmManager.load(os.path.join(dataset_directory, str(i), 'algo_manager'))
    predictions = results.get_predictions(algo_name)
    print(f"Fold {i}: {[f'{x:.4f}' for x in predictions]}")
    scores_dict[i] = predictions

# Create DataFrame
df = pd.DataFrame.from_dict(scores_dict).T
print(f"\nDataFrame:\n{df}\n")

# Compute mean and std
means = df.mean()
stds = df.std()

print("Mean ± Std across folds:")
feature_counts = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210]
for idx, fc in enumerate(feature_counts):
    print(f"{fc:3d} features: {means[idx]:.4f} ± {stds[idx]:.4f}")

print(f"\nBaseline (All features): 0.5846")
print(f"Max mean score: {means.max():.4f} at {feature_counts[means.argmax()]} features")
print(f"\nFold 4 specifically:")
print(f"Fold 4 scores: {[f'{x:.4f}' for x in scores_dict[4]]}")
print(f"Max in fold 4: {max(scores_dict[4]):.4f}")