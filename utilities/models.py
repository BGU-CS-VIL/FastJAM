# Packages
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

from utilities.homography_utils import apply_homography, get_homography_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAGEHomographyNet(nn.Module):
    def __init__(self, start_with_id_matrix, matrix_exp, in_channels=2, hidden_channels=128, out_channels=8, num_layers=5):
        super().__init__()
        self.matrix_exp = matrix_exp
        self.start_with_id_matrix = start_with_id_matrix
        self.convs = nn.ModuleList()
        # First layer: input → hidden
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        # Middle layers: hidden → hidden
        for _ in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        # Output head
        self.fc = nn.Linear(hidden_channels, out_channels)

        self._initialize_fc()
        self.to(device)

    def _initialize_fc(self):
        if self.start_with_id_matrix:
            if self.matrix_exp:
                nn.init.normal_(self.fc.weight, std=1e-5)
                nn.init.normal_(self.fc.bias, std=1e-5)
            else:
                nn.init.zeros_(self.fc.weight)
                id_params = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float, device=self.fc.bias.device)
                noise = torch.randn_like(id_params) * 1e-5  # small noise
                self.fc.bias.data.copy_(id_params + noise)

    def forward(self, data):
        assert data.edge_index.min() >= 0
        assert data.edge_index.max() < data.x.shape[0]
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out
    
    def forward_n(self, data, n):
        data = data.clone()
        device = data.x.device
        batch_size = data.batch.max().item() + 1
        
        # Initialize total transformation as identity
        H_total = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 3, 3)

        for _ in range(n):
            pred_params = self.forward(data)  # Predict delta warp
            pred_params = pred_params.clamp(min=-3.0, max=3.0)
            H_delta = get_homography_matrix(pred_params, self.matrix_exp)  # (B, 3, 3)

            # Apply current delta warp to graph.x
            new_x = []
            for i in range(batch_size):
                idx = (data.batch == i)
                x_i = data.x[idx]
                x_i_warped = apply_homography(x_i, H_delta[i])
                new_x.append(x_i_warped)
            
            data.x = torch.cat(new_x, dim=0)

            # Update total transformation
            H_total = H_delta @ H_total

        return H_total