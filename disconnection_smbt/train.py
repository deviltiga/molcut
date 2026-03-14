import torch
import os

def train(model, train_loader, criterion, optimizer, epoch, num_epochs, save_dir, device):
    model.to(device)
    model.train()

    total_loss = 0
    for batch_idx, (features, labels) in enumerate(train_loader):
        embeddings_8heads = features['emb'].to(device)
        adj_matrices = features['input'].to(device)
        cut_nums = features['cut_num'].to(device)
        output_matrices = labels.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings_8heads, adj_matrices, cut_nums)
        loss = criterion(outputs, output_matrices)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print loss every 50 steps
        if (batch_idx + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Save model only when epoch is a multiple of 100
    if (epoch + 1) % 50 == 0:
        save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_loader),
        }, save_path)
        print(f'Model saved at epoch {epoch + 1}')

    return total_loss / len(train_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            embeddings_8heads = features['emb'].to(device)
            adj_matrices = features['input'].to(device)
            cut_nums = features['cut_num'].to(device)
            output_matrices = labels.to(device)

            outputs = model(embeddings_8heads, adj_matrices, cut_nums)
            loss = criterion(outputs, output_matrices)
            total_loss += loss.item()

            # Calculate accuracy
            predicted = outputs.argmax(dim=1)
            correct = (predicted == output_matrices).sum().item()
            total_correct += correct
            total_pixels += output_matrices.numel()

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_pixels

    return avg_loss, accuracy

if __name__ == '__main__':
    # This block can be removed if not needed
    pass