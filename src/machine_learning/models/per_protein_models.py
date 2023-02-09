from torch.nn import Conv2d
import torch.nn.functional as F
import torch.nn as nn
import torch


class ProteinEmbeddingCNN(nn.Module):
    def __init__(self):
        super(ProteinEmbeddingCNN, self).__init__()
        self.conv1 = Conv2d(in_channels=1024, out_channels=32, kernel_size=(7, 1), padding=(3, 0))
        self.conv2 = Conv2d(in_channels=32, out_channels=16, kernel_size=(7, 1), padding=(3, 0))
        self.conv3 = Conv2d(in_channels=16, out_channels=8, kernel_size=(7, 1), padding=(3, 0))
        self.conv4 = Conv2d(in_channels=8, out_channels=3, kernel_size=(7, 1), padding=(3, 0))

    # Defining the forward pass
    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.squeeze(dim=-1)
        return x


class ProteinEmbeddingPssmCNN(nn.Module):
    def __init__(self):
        super(ProteinEmbeddingPssmCNN, self).__init__()
        self.conv1 = Conv2d(in_channels=1044, out_channels=32, kernel_size=(7, 1), padding=(3, 0))
        self.conv2 = Conv2d(in_channels=32, out_channels=16, kernel_size=(7, 1), padding=(3, 0))
        self.conv3 = Conv2d(in_channels=16, out_channels=8, kernel_size=(7, 1), padding=(3, 0))
        self.conv4 = Conv2d(in_channels=8, out_channels=3, kernel_size=(7, 1), padding=(3, 0))

    # Defining the forward pass
    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.squeeze(dim=-1)
        return x


class ProteinEmbeddingPssmCNNSplitInput(nn.Module):
    def __init__(self):
        super(ProteinEmbeddingPssmCNNSplitInput, self).__init__()
        self.conv1 = Conv2d(in_channels=1024, out_channels=32, kernel_size=(7, 1), padding=(3, 0))
        self.conv2 = Conv2d(in_channels=32, out_channels=16, kernel_size=(7, 1), padding=(3, 0))
        self.conv3 = Conv2d(in_channels=16, out_channels=8, kernel_size=(7, 1), padding=(3, 0))
        self.conv4 = Conv2d(in_channels=8, out_channels=3, kernel_size=(7, 1), padding=(3, 0))

        self.convPssm = Conv2d(in_channels=20, out_channels=8, kernel_size=(7, 1), padding=(3, 0))

    # Defining the forward pass
    def forward(self, x):
        embeddings = x[:, :, :1024]
        pssm = x[:, :, 1024:]

        x1 = embeddings.permute(0, 2, 1).unsqueeze(dim=-1)
        x2 = pssm.permute(0, 2, 1).unsqueeze(dim=-1)

        x1 = F.leaky_relu(self.conv1(x1))
        x1 = F.leaky_relu(self.conv2(x1))
        x1 = F.leaky_relu(self.conv3(x1))
        x2 = F.leaky_relu(self.convPssm(x2))

        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.squeeze(dim=-1)
        return x
