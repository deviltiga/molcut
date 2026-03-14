import os.path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import ast
import re
import logging
from tqdm import tqdm
from localmapper import localmapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_atom_map(atom_map_str):
    cleaned_str = atom_map_str[2:-2]
    reactants, products = cleaned_str.split('>>')
    return reactants, products


def get_bonds(smarts):
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        logger.warning(f"Could not parse SMARTS: {smarts}")
        return set()
    bonds = set()
    for bond in mol.GetBonds():
        a1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
        a2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
        if a1 and a2:
            bonds.add(tuple(sorted([a1, a2])))
    return bonds


def get_new_bonds(reactants_smarts, products_smarts):
    reactant_bonds = get_bonds(reactants_smarts)
    product_bonds = get_bonds(products_smarts)
    return list(product_bonds - reactant_bonds)


def map_to_smiles(new_bonds):
    return [(a1 - 1, a2 - 1) for a1, a2 in new_bonds]


def process_data(input_file, output_file):
    df = pd.read_csv(input_file)

    products = []
    new_bonds = []

    for index, row in df.iterrows():
        try:
            reaction_smarts = row['reactions']
            reactants_smarts, products_smarts = parse_atom_map(row['atom_map'])

            product_smiles = reaction_smarts.split('>>')[-1]
            atom_map_new_bonds = get_new_bonds(reactants_smarts, products_smarts)
            smiles_new_bonds = map_to_smiles(atom_map_new_bonds)

            products.append(product_smiles)
            new_bonds.append(str(smiles_new_bonds))

            logger.info(f"Processed row {index}: New bonds found: {atom_map_new_bonds}, "
                        f"Mapped to SMILES: {smiles_new_bonds}")
        except Exception as e:
            logger.error(f"Error processing row {index}: {e}")
            products.append("")
            new_bonds.append("[]")

    df['product'] = products
    df['new_bond'] = new_bonds

    df.to_csv(output_file, index=False)
    logger.info(f"Processed data saved to {output_file}")

def pre_process(data_path,save_path):
    device = 'cpu'
    mapper = localmapper(device)

    df = pd.read_csv(data_path)


    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            examples = {}
            examples[row['id']] = row['reactions']
            rxns = list(examples.values())

            result = mapper.get_atom_map(rxns)

            df.at[index, 'atom_map'] = str(result)


            if index % 100 == 0:
                df.to_csv(save_path, index=False)

        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            continue

    df.to_csv(save_path, index=False)

    print("Processing complete!")



if __name__ == '__main__':
    data_path=os.path.join('..', 'data', 'raw_data', 'USPTO_50K.csv')
    processed_path = os.path.join('..', 'data', 'treated_data', 'USPTO_50K_processed.csv')
    save_path = os.path.join('..', 'data', 'treated_data', 'USPTO_50K_cut.csv')

    pre_process(data_path,processed_path)
    process_data(processed_path, save_path)