import os
import csv
import time
import numpy as np



def safe_load_npy(file_path):
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Cannot load file {file_path}: {str(e)}")
        return None


def clean_and_load_data(data_folder, mapping_file,cleaned_disconnect_path):
    data_load = []
    removed_entries = []

    with open(mapping_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            npy_file_path = os.path.join(data_folder, f"{row['ID']}.npy")
            if os.path.exists(npy_file_path):
                data_load.append({'smiles': row['SMILES'], 'id': row['ID'], 'npy_file_path': npy_file_path})
            else:
                removed_entries.append(row)

    cleaned_mapping_file = cleaned_disconnect_path
    with open(cleaned_mapping_file, 'w', newline='') as csvfile:
        fieldnames = ['SMILES', 'ID']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in data_load:
            writer.writerow({'SMILES': item['smiles'], 'ID': item['id']})

    print(f"Cleaned mapping file saved to: {cleaned_mapping_file}")
    print(f"Original entries: {len(data_load) + len(removed_entries)}")
    print(f"Removed entries: {len(removed_entries)}")
    print(f"Kept entries: {len(data_load)}")

    return data_load


def main():
    emb_path_ = os.path.join('..', 'data', 'final_data', 'emb')
    disconnect_path = os.path.join('..', 'data', 'treated_data', 'disconnect.csv')
    cleaned_disconnect_path = os.path.join('..', 'data', 'treated_data', 'cleaned_disconnect.csv')

    data_load = clean_and_load_data(emb_path_, disconnect_path,cleaned_disconnect_path)

    total_files = len(data_load)
    successfully_loaded = 0
    failed_files = []

    start_time = time.time()

    for i, item in enumerate(data_load):
        try:
            npy_data = safe_load_npy(item['npy_file_path'])
            if npy_data is not None:
                successfully_loaded += 1
            else:
                failed_files.append(item['npy_file_path'])
        except Exception as e:
            print(f"Error processing file {item['npy_file_path']}: {str(e)}")
            failed_files.append(item['npy_file_path'])

        if (i + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {i + 1}/{total_files} files. Time: {elapsed_time:.2f} seconds")

    end_time = time.time()
    total_time = end_time - start_time

    print("\nTest completed!")
    print(f"Total files: {total_files}")
    print(f"Successfully loaded: {successfully_loaded}")
    print(f"Failed to load: {len(failed_files)}")
    print(f"Total time: {total_time:.2f} seconds")

    if failed_files:
        print("\nFiles that could not be loaded:")
        for file in failed_files:
            print(file)
    else:
        print("\nAll files loaded successfully!")

    success_rate = (successfully_loaded / total_files) * 100
    print(f"\nSuccess rate: {success_rate:.2f}%")


if __name__ == "__main__":
    main()
