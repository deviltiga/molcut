import numpy as np
from rdkit import Chem
import os
import pandas as pd
import ast
import re

SMILE_TOKENS = [
    'c', 'C', '(', ')', '1', 'O', '2', 'N', '=', '3', 'n', '[C@@H]',
    '4', '[C@H]', 'F', '[NH+]', 'S', 's', 'Cl', 'o', '[nH]', '[NH2+]',
    '[nH+]', '5', '[O-]', '#', 'Br', '/', '[NH3+]', '[C@@]', '[C@]',
    '\\', '[N-]', '6', '[n-]', 'I', '[N+]', '-', '[S@@]', '[S@]', '[N]',
    '[H]', '[NH]', '[NH-]', '[C]', '7', '[S+]', '[n+]', '[CH]', 'P',
    '.', '8', '[O+]', '[o+]', '[O]', '[NH2]', '9', '[CH-]', '[P]',
    '[Si]', '[C-]', '[s+]', '[OH+]', '[2H]', '%10', '[NH3]', '[N@@]',
    '[CHC@@H-2-]', '[]', '[P+]', '[S-]', '[As]', '[N@]', '%11', '[Br-]',
    '[Sn]', '[CH2]', '[I-]', '[Hg]', 'B', '[SH]', '[*]', '[PH]',
    '[Cl-]', '[S]', '[B-]', '[AlH3]', '[Se]', '[C@H-]', '%12', '[N@@+]',
    '[S@+]', '%13', '[N@+]', '[Pt]', '[se]', '[cH-]', '[P@]', '[N@@H]',
    '[S@@+]', '[Cr]', '[Cl+3]', '[I+]', '[Cu]', '[P@@]', '[Zn+2]',
    '[c+]', '[Na+]', '[Te]', '[Fe]', '[c-]', '[IH2]', '[Ba+2]', '[Cd]',
    '[Au+]', '[Zn]', '[F]', '%14', '[Bi]', '[Sb]', '[Mo]', '[Cu+2]',
    'p', '[N@H]', '[Cl]', '[Au]', '[In]', '[Pt+2]', '[Pd]', '[B]',
    '[N@H+]', '[3H]', '[NH4+]', '[Ca+2]', '[K+]', '[Hg+2]', '[Fe+2]',
    '[Co+2]', '[Fe+3]', '[SiH]', '[Ge]', '[Ag]', '[SH-]', '[Co]',
    '[Cu+]', '[V]', '[C@-]', '[13C]', '[H+]', '[Mg+2]', '[Zr]', '[Na]',
    '[Gd+3]', '[Co+]', '[Li+]', '[Ni+2]', '[Mn+2]', '[Ti]', '[Ni]',
    '[GaH3]', '[Ac]', '[BiH3]', '[Mo-2]', '[pH]', '[N++]', '[Br]',
    '[15N]', '[OH-]', '[TlH2+]', '[Ba]', '[Ag+]', '[Cr+3]', '[Nd]',
    '[Yb]', '[PbH2+2]', '[Cd+2]', '[SnH2+2]', '[Ti+2]', '[Dy]', '[Ca]',
    '[Sr+2]', '[Be+2]', '[Cr+2]', '[Mn+]', '[SbH6+3]', '[Au-]', '[Fe-]',
    '[Fe-2]',
    '[U]'
]

ATOM_TOKENS = [
    'c', 'C', 'O', 'N', 'n', '[C@@H]',
    '[C@H]', 'F', '[NH+]', 'S', 's', 'Cl', 'o', '[nH]', '[NH2+]',
    '[nH+]', '[O-]', '#', 'Br', '[NH3+]', '[C@@]', '[C@]',
    '[N-]', '[n-]', 'I', '[N+]', '[S@@]', '[S@]', '[N]',
    '[H]', '[NH]', '[NH-]', '[C]', '[S+]', '[n+]', '[CH]', 'P',
    '[O+]', '[o+]', '[O]', '[NH2]', '[CH-]', '[P]',
    '[Si]', '[C-]', '[s+]', '[OH+]', '[2H]', '[NH3]', '[N@@]',
    '[CH2-]', '[C@@H-]', '[P+]', '[S-]', '[As]', '[N@]', '[Br-]',
    '[Sn]', '[CH2]', '[I-]', '[Hg]', 'B', '[SH]', '[PH]',
    '[Cl-]', '[S]', '[B-]', '[AlH3]', '[Se]', '[C@H-]', '[N@@+]',
    '[S@+]', '[N@+]', '[Pt]', '[se]', '[cH-]', '[P@]', '[N@@H]',
    '[S@@+]', '[Cr]', '[Cl+3]', '[I+]', '[Cu]', '[P@@]', '[Zn+2]',
    '[c+]', '[Na+]', '[Te]', '[Fe]', '[c-]', '[IH2]', '[Ba+2]', '[Cd]',
    '[Au+]', '[Zn]', '[F]', '[Bi]', '[Sb]', '[Mo]', '[Cu+2]',
    'p', '[N@H]', '[Cl]', '[Au]', '[In]', '[Pt+2]', '[Pd]', '[B]',
    '[N@H+]', '[3H]', '[NH4+]', '[Ca+2]', '[K+]', '[Hg+2]', '[Fe+2]',
    '[Co+2]', '[Fe+3]', '[SiH]', '[Ge]', '[Ag]', '[SH-]', '[Co]',
    '[Cu+]', '[V]', '[C@-]', '[13C]', '[H+]', '[Mg+2]', '[Zr]', '[Na]',
    '[Gd+3]', '[Co+]', '[Li+]', '[Ni+2]', '[Mn+2]', '[Ti]', '[Ni]',
    '[GaH3]', '[Ac]', '[BiH3]', '[Mo-2]', '[pH]', '[N++]', '[Br]',
    '[15N]', '[OH-]', '[TlH2+]', '[Ba]', '[Ag+]', '[Cr+3]', '[Nd]',
    '[Yb]', '[PbH2+2]', '[Cd+2]', '[SnH2+2]', '[Ti+2]', '[Dy]', '[Ca]',
    '[Sr+2]', '[Be+2]', '[Cr+2]', '[Mn+]', '[SbH6+3]', '[Au-]', '[Fe-]',
    '[Fe-2]',
    '[U]'
]

SORTED_SMILE_TOKENS = sorted(SMILE_TOKENS, key=len, reverse=True)

PATTERN = '|'.join(re.escape(token) for token in SORTED_SMILE_TOKENS)

def tokenize_smiles(smiles):
    tokens = re.findall(PATTERN, smiles)
    return tokens


def update_bond_positions(smiles, original_positions):
    tokens = tokenize_smiles(smiles)

    atom_to_token = {}
    atom_index = 0
    for i, token in enumerate(tokens):
        if is_atom(token):
            atom_to_token[atom_index] = i
            atom_index += 1

    updated_positions = []
    for start, end in original_positions:
        if start in atom_to_token and end in atom_to_token:
            updated_positions.append((atom_to_token[start], atom_to_token[end]))
        else:
            print(f"Warning: Could not update position for bond {start}-{end}")

    return updated_positions


def is_atom(token):
    if token in ATOM_TOKENS:
        return True
    return False


def create_adjacency_matrices(smiles, label):
    tokens = tokenize_smiles(smiles)
    n = len(tokens)

    input_matrix = np.zeros((n, n), dtype=int)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    atom_to_token = {}
    token_to_atom = {}
    atom_index = 0
    for i, token in enumerate(tokens):
        if is_atom(token):
            atom_to_token[atom_index] = i
            token_to_atom[i] = atom_index
            atom_index += 1

    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        if begin_atom in atom_to_token and end_atom in atom_to_token:
            start = atom_to_token[begin_atom]
            end = atom_to_token[end_atom]
            input_matrix[start, end] = input_matrix[end, start] = 2
        else:
            print(f"Warning: Could not map bond {begin_atom} - {end_atom} to tokens")
            print(tokens)

    for i in range(n):
        if not is_atom(tokens[i]):
            input_matrix[i, :] = input_matrix[:, i] = 1

    output_matrix = input_matrix.copy()

    for start, end in label:
        if 0 <= start < n and 0 <= end < n:
            output_matrix[start, end] = output_matrix[end, start] = 3
        else:
            print(f"Warning: Label {start} - {end} is out of bounds")

    return input_matrix, output_matrix, tokens

def save_matrix(smiles_id_mapping_path,input_matrix_path_,output_matrix_path_):
    df = pd.read_csv(smiles_id_mapping_path)
    df['cut_bond'] = df['cut_bond'].apply(ast.literal_eval)

    for index, row in df.iterrows():
        smiles = row['SMILES']
        label = row['cut_bond']
        input_matrix, output_matrix, tokens = create_adjacency_matrices(smiles, label)

        input_matrix_path = os.path.join(input_matrix_path_, f"{row['ID']}.npy")
        np.save(input_matrix_path, input_matrix)
        output_matrix_path = os.path.join(output_matrix_path_, f"{row['ID']}.npy")
        np.save(output_matrix_path, output_matrix)
        if index % 100 == 0:
            print(index)
            print(len(df))
            print(f'{(index+1)/ len(df) * 100:.2f}%')
            print("----------------------------------------------------------")

    print("100.00")
    print("Finished")


if __name__ == '__main__':

    smiles_id_mapping_path = os.path.join('..', 'data', 'final_data', 'smiles_id_mapping.csv')
    input_matrix_path_ = os.path.join('..', 'data', 'final_data', 'input_matrix')
    output_matrix_path_ = os.path.join('..', 'data', 'final_data', 'output_matrix')
    embedding_path_ = os.path.join('..', 'data', 'final_data', 'emb')

    df = pd.read_csv(smiles_id_mapping_path)
    df = df[1:3]
    df['cut_bond'] = df['cut_bond'].apply(ast.literal_eval)

    for index, row in df.iterrows():
        smiles = row['SMILES']
        label = row['cut_bond']
        input_matrix, output_matrix, tokens = create_adjacency_matrices(smiles, label)

        input_matrix_path = os.path.join(input_matrix_path_, f"{row['ID']}.npy")
        output_matrix_path = os.path.join(output_matrix_path_, f"{row['ID']}.npy")
        embedding_path = os.path.join(embedding_path_, f"{row['ID']}.npy")
        embedding = np.load(embedding_path)
        print('smiles: ', smiles)
        print('smiles length: ', len(smiles))
        print('tokens: ', tokens)
        print('embedding shape: ', embedding.shape)
        print('input_matrix shape: ', input_matrix.shape)
