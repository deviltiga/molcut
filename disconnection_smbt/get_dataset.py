import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np

class Molcutset(Dataset):
    def __init__(self, df, emb_path_, input_matrix_path_, output_matrix_path_, max_token_size=160):
        self.ids = df['ID']
        self.cut_bonds_nums = df['cut_bond_num']
        self.emb_path_ = emb_path_
        self.input_matrix_path_ = input_matrix_path_
        self.output_matrix_path_ = output_matrix_path_
        self.max_token_size = max_token_size

    def __len__(self):
        return len(self.ids)

    def get_file(self, path_, id):
        id += '.npy'
        path = os.path.join(path_, id)
        return np.load(path)

    def preprocess_emb(self, emb):
        # Select the last layer (8 heads)
        last_layer_emb = emb[:, -1, :, :]  # Shape: (8, token, token)
        
        # Pad or truncate to max_token_size
        padded_emb = np.zeros((8, self.max_token_size, self.max_token_size))
        token_size = min(last_layer_emb.shape[1], self.max_token_size)
        padded_emb[:, :token_size, :token_size] = last_layer_emb[:, :token_size, :token_size]
        
        return padded_emb

    def preprocess_matrix(self, matrix):
        # Pad or truncate matrix to max_token_size
        padded_matrix = np.zeros((self.max_token_size, self.max_token_size))
        token_size = min(matrix.shape[0], self.max_token_size)
        padded_matrix[:token_size, :token_size] = matrix[:token_size, :token_size]
        
        return padded_matrix

    def __getitem__(self, index):
        input_matrix = self.get_file(path_=self.input_matrix_path_, id=self.ids[index])
        embedding = self.get_file(path_=self.emb_path_, id=self.ids[index])
        output_matrix = self.get_file(path_=self.output_matrix_path_, id=self.ids[index])

        # Preprocess data
        input_matrix = self.preprocess_matrix(input_matrix)
        embedding = self.preprocess_emb(embedding)
        output_matrix = self.preprocess_matrix(output_matrix)

        feature = {
            'emb': torch.FloatTensor(embedding),
            'input': torch.FloatTensor(input_matrix),
            'cut_num': self.cut_bonds_nums[index]
        }
        label = torch.FloatTensor(output_matrix)
        return feature, label

def collate_fn(batch):
    # Custom collate function to handle batches
    features = {
        'emb': torch.stack([item[0]['emb'] for item in batch]),
        'input': torch.stack([item[0]['input'] for item in batch]),
        'cut_num': torch.tensor([item[0]['cut_num'] for item in batch])
    }
    labels = torch.stack([item[1] for item in batch])
    return features, labels

if __name__ == '__main__':
    smiles_id_mapping_path = os.path.join('.', 'data', 'final_data', 'smiles_id_mapping.csv')
    emb_path_ = os.path.join('.', 'data', 'final_data', 'emb')
    input_matrix_path_ = os.path.join('.', 'data', 'final_data', 'input_matrix')
    output_matrix_path_ = os.path.join('.', 'data', 'final_data', 'output_matrix')

    df = pd.read_csv(smiles_id_mapping_path)
    dataset = Molcutset(df, emb_path_, input_matrix_path_, output_matrix_path_)

    # Test the dataset
    sample_feature, sample_label = dataset[0]
    print("Sample feature shapes:")
    print("Embedding shape:", sample_feature['emb'].shape)
    print("Input matrix shape:", sample_feature['input'].shape)
    print("Label shape:", sample_label.shape)