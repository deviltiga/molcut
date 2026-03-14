import os
import csv
import hashlib
from fairseq.models.roberta import RobertaModel
import argparse
import sys
import numpy as np
import torch



folder_name_smiles = os.path.join('..', 'data', 'test', 'disconnect_USPTO50K.smi')
disconnect_path_ = os.path.join('..', 'data', 'test')
emb_path_ = os.path.join('..', 'data', 'test', 'emb')


def create_smiles_id_mapping(input_file, disconnect_path_):
    os.makedirs(disconnect_path_, exist_ok=True)
    mapping_file = os.path.join(disconnect_path_, 'disconnect.csv')
    smiles_id_map = {}

    with open(input_file, 'r') as f:
        for line in f:
            smiles = line.strip()
            smiles_id = hashlib.md5(smiles.encode()).hexdigest()[:10]  # Use first 10 characters of MD5 hash
            smiles_id_map[smiles] = smiles_id

    with open(mapping_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SMILES', 'ID'])
        for smiles, id in smiles_id_map.items():
            writer.writerow([smiles, id])

    return smiles_id_map


def load_pretrain_model(model_name_or_path, checkpoint_file, data_name_or_path, bpe='smi'):
    pretrain_model = RobertaModel.from_pretrained(
        model_name_or_path,
        checkpoint_file,
        data_name_or_path,
        bpe=bpe,
    )
    pretrain_model.eval()
    return pretrain_model


def extract_hidden(pretrain_model, target_file, smiles_id_map, emb_path_):
    sample_num = sum(1 for line in open(target_file) if line.strip())
    hidden_features = {i: None for i in range(sample_num)}

    for i, line in enumerate(open(target_file)):
        if not line.strip():
            continue

        smiles = line.strip()
        smiles_id = smiles_id_map.get(smiles, f"unknown_{i}")

        tokens = pretrain_model.encode(smiles)
        print(tokens)
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat((tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))

        _, all_layer_hiddens, attn, attn_8head = pretrain_model.model(
            tokens.unsqueeze(0), features_only=True, return_all_hiddens=True)

        hidden_info = all_layer_hiddens['inner_states'][-1]
        hidden_features[i] = hidden_info.squeeze(1).cpu().detach().numpy()

        data_files_npy = os.path.join(emb_path_, f'{smiles_id}.npy')
        attn_8head = np.array(attn_8head).squeeze(1)
        np.save(data_files_npy, attn_8head)

    return hidden_features, attn, attn_8head


def extract_features_from_hidden(hidden_info):
    samples_num = len(hidden_info)
    hidden_dim = np.shape(hidden_info[0])[-1]
    samples_features = np.zeros([samples_num, hidden_dim])
    for n_sample, hidden in hidden_info.items():
        samples_features[n_sample, :] = hidden[0, :]
    return samples_features


def main(args):
    # Create SMILES-ID mapping
    smiles_id_map = create_smiles_id_mapping(args.target_file, disconnect_path_)

    pretrain_model = load_pretrain_model(
        args.model_name_or_path, args.checkpoint_file, args.data_name_or_path, args.bpe)

    hidden_info, attn, attn_8head = extract_hidden(pretrain_model, args.target_file, smiles_id_map, emb_path_)
    attn_8head = np.array(attn_8head)

    print('Generate features from hidden information')
    samples_features = extract_features_from_hidden(hidden_info)
    print(f'Features shape: {np.shape(samples_features)}')
    #np.save(os.path.join(emb_path_, 'extract_f1.npy'), samples_features)
    return attn_8head


def parse_args(args):
    parser = argparse.ArgumentParser(description="Tools kit for downstream jobs")

    parser.add_argument('--model_name_or_path', default="./chembl_pubchem_zinc_models/chembl27_512/", type=str,
                        help='Pretrained model folder')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt', type=str,
                        help='Pretrained model name')
    parser.add_argument('--data_name_or_path', default="./chembl_pubchem_zinc_models/chembl27_512/", type=str,
                        help="Pre-training dataset folder")
    parser.add_argument('--dict_file', default='dict.txt', type=str,
                        help="Pre-training dict filename(full path)")
    parser.add_argument('--bpe', default='smi', type=str)
    parser.add_argument('--target_file', default=folder_name_smiles, type=str,
                        help="Target file for feature extraction, default format is .smi")
    args = parser.parse_args()
    return args


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    attn_twodim_array = main(args)
    return attn_twodim_array


if __name__ == '__main__':
    attn = cli_main()

