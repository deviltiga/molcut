# Disconnection SMBT Model Training

## Data Processing Pipeline

### 1. Raw Data → Cut Bond Data
```
USPTO_50K.csv → generate_cut_bond.py → USPTO_50k_cut.csv
```
- Input: `data/raw_data/USPTO_50K.csv`
- Output: `data/treated_data/USPTO_50k_cut.csv`
- Description: Identify cut bond positions in chemical reactions

### 2. Cut Bond Data → SMILES Representation
```
USPTO_50k_cut.csv → generate_smi.py → disconnect_USPTO50K.smi
```
- Input: `data/treated_data/USPTO_50k_cut.csv`
- Output: `data/treated_data/disconnect_USPTO50K.smi`

### 3. SMILES → Embedding
```
disconnect_USPTO50K.smi → generate_disconnect.py → emb/ + disconnect.csv
```
- Input: `data/treated_data/disconnect_USPTO50K.smi`
- Output: `data/final_data/emb/` (embedding vectors) + `data/treated_data/disconnect.csv`

### 4. Clean Disconnect Data
```
disconnect.csv → clean_disconect.py → cleaned_disconnect.csv
```
- Input: `data/treated_data/disconnect.csv`
- Output: `data/treated_data/cleaned_disconnect.csv`

### 5. Generate SMILES-ID Mapping
```
USPTO_50k_cut.csv + cleaned_disconnect.csv → smiles_id_mapping.csv
```
- Input: `data/treated_data/USPTO_50k_cut.csv` + `data/treated_data/cleaned_disconnect.csv`
- Output: `data/final_data/smiles_id_mapping.csv`

### 6. Generate Adjacency Matrix
```
smiles_id_mapping.csv → generate_matrix.py → input_matrix/ + output_matrix/
```
- Input: `data/final_data/smiles_id_mapping.csv`
- Output: `data/final_data/input_matrix/` + `data/final_data/output_matrix/`

## Model Training

### Run Training

```bash
python main.py
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| batch_size | 16 | Batch size |
| num_epochs | 2000 | Number of training epochs |
| learning_rate | 0.0001 | Learning rate |
| custom_weights | [1.0, 1.0, 1.0, 1.0] | Loss function weights |

### Data Loading

- `smiles_id_mapping.csv`: SMILES-ID mapping file
- `emb/`: Embedding vectors directory
- `input_matrix/`: Input adjacency matrix directory
- `output_matrix/`: Output adjacency matrix directory

### Dataset Split

- Training set: 80%
- Test set: 20%

### Output

- Model saved in: `model_unet_lr0001_wt1/` directory
- Training log: `model_unet_lr0001_wt1/training_log.csv` (includes Epoch, Train Loss, Test Loss, Test Accuracy)

## Environment Requirements

- `generate_cut_bond.py` requires localmapper environment
