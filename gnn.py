import torch.nn.functional as F
import torch.nn
from torch_geometric.nn import GATv2Conv, Linear, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(512, 128, edge_dim=512)
        self.conv2 = GATv2Conv(128, 32, edge_dim=512)
        self.conv3 = GATv2Conv(32, 8, edge_dim=512)
        self.lin = Linear(8, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr).relu()
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)
        return self.lin(x).sigmoid()
