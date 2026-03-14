import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100):
        super(SegmentationLoss, self).__init__()
        if weight is None:
            self.weight = torch.tensor([1.0, 1.0, 1.0, 3.0])
        else:
            self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_classes, height, width)
        targets: (batch_size, height, width)
        """
        self.weight = self.weight.to(inputs.device)
        
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        
        
        targets = targets.view(targets.size(0), -1).long()
        
        loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        return loss

if __name__ == "__main__":
    inputs = torch.randn(2, 4, 16, 16)
    targets = torch.randint(0, 4, (2, 16, 16))
    
    criterion = SegmentationLoss()
    loss = criterion(inputs, targets)
    print(f"Loss: {loss.item()}")
    
    custom_weights = torch.tensor([1.0, 1.0, 1.0, 3.0])
    criterion_custom = SegmentationLoss(weight=custom_weights)
    loss_custom = criterion_custom(inputs, targets)
    print(f"Loss with custom weights: {loss_custom.item()}")
