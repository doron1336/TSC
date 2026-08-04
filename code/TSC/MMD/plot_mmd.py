# Modified plotting code
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import paths  # noqa: F401  # TSC path config (TSC_CODE_DIR, TSC_DATA_DIR, TSC_RESULTS_DIR)

ucr_dir = paths.UCR_DIR
har_dataset = 'HAR_datasets'
without_GA = 'without_GA'
HAR_DATASETS = ['GestureMidAirD3', 'GestureMidAirD2', 'UWaveGestureLibraryAll', 'GesturePebbleZ2', 'AllGestureWiimoteX',
                'CricketX', 'CricketY']
percentages = np.arange(10, 101, 10)  # 10%, ..., 100%

DIRECTORY_NUM: int = 3
for dataset_name in HAR_DATASETS:
    without_har = os.path.join(ucr_dir, dataset_name)
    base_dir = os.path.join(ucr_dir, har_dataset, dataset_name)
    directory = os.path.join(base_dir, str(DIRECTORY_NUM))
    mmd_dir = os.path.join(directory, 'MMD')
    # Load the saved .pkl file
    pkl_path = os.path.join(mmd_dir, 'arrays.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Extract arrays from loaded data
    prediction_score_trimmed = data["prediction_score_trimmed"]
    prediction_score_selected = data["prediction_score_selected"]
    mmd_result = data["mmd_result"]

    # Create plot with two y-axes and solid lines
    fig, ax1 = plt.subplots()

    # Plot MMD score on the left y-axis with solid line
    ax1.plot(percentages, mmd_result, '-o', label='MMD Score', color='red', markersize=5)
    ax1.set_xlabel('Percentage of Signal Kept (%)')
    ax1.set_ylabel('MMD Score', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    # Add grid to the first axis
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Create second y-axis for prediction scores
    ax2 = ax1.twinx()
    ax2.plot(percentages, prediction_score_trimmed, '-^', label='Prediction Score (Trimmed)', color='blue',
             markersize=5)
    ax2.plot(percentages, prediction_score_selected, '-s', label='Prediction Score (Selected)', color='green',
             markersize=5)
    ax2.set_ylabel('Prediction Score', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Add title
    plt.title('MMD and Prediction Scores vs Percentage of Signal Kept')

    # Combine legends from both axes and place below the plot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), borderaxespad=0.,
               ncol=3)

    # Adjust layout to prevent overlap and accommodate legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin to prevent legend cutoff

    output_path = os.path.join(paths.FIGURES_DIR, f'{dataset_name}_plot.png')
    plt.savefig(output_path)  # Save as PNG (or use .pdf, .svg, etc.)

    # Show plot
    plt.show()
    plt.close()