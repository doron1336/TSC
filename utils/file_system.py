import os


def find_subfolders_with_file(root_folder: str, target_subfolder: str, target_file: str) -> str:
    for root, dirs, files in os.walk(root_folder):
        # Check if we're in the target subfolder
        if os.path.basename(root) == target_subfolder:
            if target_file in files:
                # Add the parent folder of the target subfolder to the result list
                return root_folder.split("/")[-1]
