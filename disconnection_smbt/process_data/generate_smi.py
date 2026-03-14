import os
import pandas as pd

def save_smi_without_header(df, column_name, output_path):
    with open(output_path, 'w') as f:
        for item in df[column_name]:
            f.write(f"{item}\n")
    print(f"Successfully generated {output_path} without header")

if __name__ == '__main__':
    data_path = os.path.join('..', 'data', 'treated_data', 'USPTO_50K_cut.csv')
    smi_path = os.path.join('..', 'data', 'treated_data', 'disconnect_USPTO50K.smi')

    if not os.path.exists(data_path):
        print(f"Error: File not found: {data_path}")
        print(f"Current working directory: {os.getcwd()}")
        exit(1)

    df = pd.read_csv(data_path)
    save_smi_without_header(df, 'product', smi_path)
