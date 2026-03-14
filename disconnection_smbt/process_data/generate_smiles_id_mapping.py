import os
import pandas as pd
import glob
import ast


def load_cleaned_mapping(file_path):
    return pd.read_csv(file_path)


def load_output_csvs(directory):
    '''all_data = pd.DataFrame()
    for csv_file in glob.glob(os.path.join(directory, '*.csv')):
        df = pd.read_csv(csv_file)
        all_data = pd.concat([all_data, df], ignore_index=True)
        return all_data'''
    df = pd.read_csv(directory)
    return df


def update_mapping_with_cut_bond(cleaned_mapping, output_data):
    # Create a dictionary for faster lookup
    output_dict = dict(zip(output_data['product'], output_data['new_bond']))

    # Add new column 'cut_bond'
    cleaned_mapping['cut_bond'] = cleaned_mapping['SMILES'].map(output_dict)
    print(cleaned_mapping['cut_bond'].head())

    return cleaned_mapping


def main():
    # File paths
    cleaned_mapping_path = os.path.join('..', 'data', 'treated_data', 'cleaned_disconnect.csv')
    output_csv_directory = os.path.join('..', 'data', 'treated_data', 'USPTO_50K_cut.csv')
    smiles_id_mapping_path = os.path.join('..', 'data', 'final_data', 'smiles_id_mapping.csv')



    # Load data
    print("Loading cleaned mapping file...")
    cleaned_mapping = load_cleaned_mapping(cleaned_mapping_path)

    print("Loading output CSV files...")
    output_data = load_output_csvs(output_csv_directory)

    # Update mapping with cut_bond information
    print("Updating mapping with cut_bond information...")
    updated_mapping = update_mapping_with_cut_bond(cleaned_mapping, output_data)
    print(updated_mapping.head())


    updated_mapping['cut_bond']=updated_mapping['cut_bond'].apply(ast.literal_eval)
    updated_mapping=updated_mapping[updated_mapping['cut_bond'].apply(lambda x:x!=[])].reset_index(drop=True)
    updated_mapping['cut_bond_num']=updated_mapping['cut_bond'].apply(lambda x:len(x))

    # Save updated mapping
    print(updated_mapping.head(5))

    updated_mapping.to_csv(smiles_id_mapping_path, index=False)
    print(f"Updated mapping saved to: {smiles_id_mapping_path}")

    # Print some statistics
    total_rows = len(updated_mapping)
    matched_rows = updated_mapping['cut_bond'].notna().sum()
    print(f"Total rows in cleaned mapping: {total_rows}")
    print(f"Rows matched with cut_bond: {matched_rows}")
    print(f"Match rate: {matched_rows / total_rows:.2%}")


if __name__ == "__main__":
    main()