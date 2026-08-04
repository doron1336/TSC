import os
import pickle


def find_subfolders_with_file(root_folder: str, target_subfolder: str, target_file: str) -> str:
    for root, dirs, files in os.walk(root_folder):
        # Check if we're in the target subfolder
        if os.path.basename(root) == target_subfolder:
            if target_file in files:
                # Add the parent folder of the target subfolder to the result list
                return root_folder.split("/")[-1]


def save_to_pickle(data, directory: str, file_name: str) -> str:
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Construct the full file path
    filepath = os.path.join(directory, file_name)

    # Save the data to a pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {filepath}")
    return filepath


# Example usage:
if __name__ == "__main__":
    results = {"calculation": [1, 2, 3], "status": "success"}  # Example data
    directory = "./results"  # Directory to save the pickle
    filename = "calculation_results.pkl"  # Pickle file name

    save_to_pickle(results, directory, filename)
