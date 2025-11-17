import os
import shutil

def organize_dataset(dataset_folder):
    """
    Creates new folders 't1', 't2', ..., 'tn' inside the dataset folder.
    Inside each 'ti' folder, it copies the content of 4 distinct folders
    from the dataset folder. After copying into all 'ti' folders,
    it deletes the original 4 copied folders from the dataset folder.

    Args:
        dataset_folder (str): The path to the main dataset folder.
    """
    if not os.path.isdir(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' not found.")
        return

    # Get a list of all folders directly inside the dataset folder
    all_folders = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    num_original_folders = len(all_folders)

    if num_original_folders < 4:
        print("Error: Less than 4 folders found in the dataset folder. Cannot proceed.")
        return

    num_new_folders = num_original_folders  # Create as many 't' folders as original folders

    copied_source_folders = set()  # Keep track of folders copied into 't' folders

    for i in range(1, num_new_folders + 1):
        new_folder_name = f"t{i}"
        new_folder_path = os.path.join(dataset_folder, new_folder_name)

        # Create the new 'ti' folder
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Created folder: {new_folder_path}")

        # Select 4 distinct folders to copy (excluding the current 'ti' folder's equivalent)
        folders_to_copy = []
        count = 0
        for j, folder in enumerate(all_folders):
            if count < 4 and j != (i - 1) and folder not in copied_source_folders:
                folders_to_copy.append(folder)
                count += 1
            elif count == 4:
                break

        if len(folders_to_copy) < 4:
            print(f"Warning: Could not find 4 distinct, uncopied folders to copy into {new_folder_path}.")
            # Clean up the newly created folder if not enough source folders
            os.rmdir(new_folder_path)
            continue

        print(f"Copying folders {folders_to_copy} into {new_folder_path}")
        for folder_name in folders_to_copy:
            source_path = os.path.join(dataset_folder, folder_name)
            destination_path = os.path.join(new_folder_path, folder_name)
            try:
                shutil.copytree(source_path, destination_path)
                copied_source_folders.add(folder_name)
                print(f"  - Copied '{folder_name}'")
            except Exception as e:
                print(f"  - Error copying '{folder_name}': {e}")

    # Delete the original copied folders from the dataset folder
    print("\nDeleting the originally copied folders from the dataset folder:")
    for folder_to_delete in copied_source_folders:
        folder_path_to_delete = os.path.join(dataset_folder, folder_to_delete)
        try:
            shutil.rmtree(folder_path_to_delete)
            print(f"  - Deleted '{folder_to_delete}' from '{dataset_folder}'")
        except Exception as e:
            print(f"  - Error deleting '{folder_to_delete}' from '{dataset_folder}': {e}")

if __name__ == "__main__":
    dataset_folder = "data/dataset"  # Replace "dataset" with the actual path to your dataset folder

    # Create a dummy dataset folder with some subfolders for testing
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        for i in range(1, 7):
            os.makedirs(os.path.join(dataset_folder, f"folder_{i}"))
            with open(os.path.join(dataset_folder, f"folder_{i}", "test.txt"), "w") as f:
                f.write(f"Content of folder {i}")

    organize_dataset(dataset_folder)

    # Optional: Clean up the dummy dataset folder after testing
    # shutil.rmtree(dataset_folder)