import torch
from torch.utils.data import DataLoader, random_split
from get_dataset import Molcutset, collate_fn
from loss import SegmentationLoss
from model import SurgicalToolSegmentationModel
from train import train, test
import os
import pandas as pd
import csv

if __name__ == "__main__":
    # Set parameters
    batch_size = 16
    num_epochs = 2000
    learning_rate = 0.0001
    custom_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    save_dir = 'model_unet_lr0001_wt1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Create directory to save models
    os.makedirs(save_dir, exist_ok=True)

    # Initialize model, loss function, and optimizer
    model = SurgicalToolSegmentationModel()
    criterion = SegmentationLoss(weight=custom_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load dataset
    smiles_id_mapping_path = os.path.join('.', 'data', 'final_data', 'smiles_id_mapping.csv')
    emb_path_ = os.path.join('.', 'data', 'final_data', 'emb')
    input_matrix_path_ = os.path.join('.', 'data', 'final_data', 'input_matrix')
    output_matrix_path_ = os.path.join('.', 'data', 'final_data', 'output_matrix')
    df = pd.read_csv(smiles_id_mapping_path)

    dataset = Molcutset(df, emb_path_, input_matrix_path_, output_matrix_path_)

    # Split dataset into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Use a generator with the set seed for reproducible splits
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    print('Length of train dataset:', len(train_dataset))
    print('Length of test dataset:', len(test_dataset))

    # Prepare CSV file for logging losses and accuracy
    csv_path = os.path.join(save_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])

    # Train and test the model
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, epoch, num_epochs, save_dir, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        
        # Log losses and accuracy to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch + 1, train_loss, test_loss, test_accuracy])
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    print('Training and testing completed.')