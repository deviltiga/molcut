import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SurgicalToolSegmentationModel(nn.Module):
    def __init__(self):
        super(SurgicalToolSegmentationModel, self).__init__()

        self.conv_emb = nn.Conv2d(8, 64, kernel_size=3, stride=2, padding=1)
        self.conv_adj = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)

        self.enc_block1 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.enc_block2 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            nn.Conv2d(256, 256, kernel_size=1)
        )
        self.enc_block3 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            nn.Conv2d(512, 512, kernel_size=1)
        )

        self.upper_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.upper_conv2 = nn.Conv2d(129, 128, kernel_size=3, stride=2, padding=1)
        self.upper_conv3 = nn.Conv2d(257, 256, kernel_size=3, stride=2, padding=1)

        self.fusion_conv1 = nn.Conv2d(192, 128, kernel_size=1)
        self.fusion_conv2 = nn.Conv2d(384, 256, kernel_size=1)
        self.fusion_conv3 = nn.Conv2d(768, 512, kernel_size=1)

        self.dec_block1 = nn.ConvTranspose2d(513, 256, kernel_size=4, stride=2, padding=1)
        self.dec_block2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)  
        self.dec_block3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)  
        self.dec_block4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)


        self.output_conv = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, embeddings_8heads, adj_matrices, cut_nums):
        x_emb = F.relu(self.conv_emb(embeddings_8heads))
        adj_matrices = adj_matrices.unsqueeze(1) 
        x_adj = F.relu(self.conv_adj(adj_matrices))

        x_emb = self.enc_block1(x_emb)
        x_adj = self.upper_conv1(x_adj)
        fused = torch.cat([x_emb, x_adj], dim=1)
        fused = self.fusion_conv1(fused)
        fused_128 = fused.clone()
        cut_nums_expanded = cut_nums.view(-1, 1, 1, 1).expand(-1, 1, fused.size(2), fused.size(3))
        fused = torch.cat([fused, cut_nums_expanded], dim=1)

        x_emb = self.enc_block2(x_emb)
        x_adj = self.upper_conv2(fused)
        fused = torch.cat([x_emb, x_adj], dim=1)
        fused = self.fusion_conv2(fused)
        fused_256 = fused.clone()
        cut_nums_expanded = cut_nums.view(-1, 1, 1, 1).expand(-1, 1, fused.size(2), fused.size(3))
        fused = torch.cat([fused, cut_nums_expanded], dim=1)

        x_emb = self.enc_block3(x_emb)
        x_adj = self.upper_conv3(fused)
        fused = torch.cat([x_emb, x_adj], dim=1)
        fused = self.fusion_conv3(fused)
        fused_512 = fused.clone()
        cut_nums_expanded = cut_nums.view(-1, 1, 1, 1).expand(-1, 1, fused.size(2), fused.size(3))
        fused = torch.cat([fused, cut_nums_expanded], dim=1)

        x = F.relu(self.dec_block1(fused))
        x = torch.cat([x, fused_256], dim=1)

        x = F.relu(self.dec_block2(x))
        x = torch.cat([x, fused_128], dim=1)

        x = F.relu(self.dec_block3(x))

        x = F.relu(self.dec_block4(x))

        output = self.output_conv(x)
        return output
    
if __name__ == '__main__':
    model = SurgicalToolSegmentationModel()
    print(model(torch.randn(4, 8, 160, 160), torch.randn(4, 160, 160), torch.randn(4, 1)).shape)
