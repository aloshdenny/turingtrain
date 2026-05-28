"""
Graph Neural Network for molecular and mixture encoding.
Split-head architecture with learnable scale parameter.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.nn import Linear, ReLU, Sequential, Dropout


class AtomEncoder(nn.Module):
    """Embed atom features."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(119, hidden_dim),  # atomic number (0-118)
            nn.Embedding(2, hidden_dim),    # aromatic (0-1)
            nn.Embedding(5, hidden_dim),    # degree (0-4)
            nn.Embedding(5, hidden_dim),    # formal charge + 2 (-2 to +2 mapped to 0-4)
            nn.Embedding(4, hidden_dim),    # hybridization (SP, SP2, SP3, OTHER)
            nn.Embedding(5, hidden_dim),    # num_h (0-4)
        ])

    def forward(self, x):
        # x shape: (num_atoms, 6)
        out = sum(emb(x[:, i].long()) for i, emb in enumerate(self.embeddings))
        return out


class BondEncoder(nn.Module):
    """Embed bond features."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(4, hidden_dim),  # bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC)
            nn.Embedding(2, hidden_dim),  # is aromatic
            nn.Embedding(2, hidden_dim),  # is conjugated
        ])

    def forward(self, x):
        # x shape: (num_bonds, 3)
        out = sum(emb(x[:, i].long()) for i, emb in enumerate(self.embeddings))
        return out


class MessagePassingLayer(nn.Module):
    """Single message-passing layer."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.message_mlp = Sequential(
            Linear(3 * hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: (num_atoms, hidden_dim)
        # edge_index: (2, num_edges)
        # edge_attr: (num_edges, hidden_dim)
        
        src, dst = edge_index
        
        # Message: concatenate source, destination, edge attributes
        msg_input = torch.cat([x[src], x[dst], edge_attr], dim=1)
        messages = self.message_mlp(msg_input)
        
        # Aggregate messages
        aggregated = torch.zeros_like(x)
        aggregated.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]), messages)
        
        # Update
        update_input = torch.cat([x, aggregated], dim=1)
        x_new = self.update_mlp(update_input)
        
        return x_new


class MolecularGNN(nn.Module):
    """Encode individual molecules using GNN."""
    def __init__(self, hidden_dim=64, num_layers=5):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        # CN prediction head (Softplus + learnable scale)
        self.cn_head = Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # Interaction prediction head (Tanh)
        self.interaction_head = Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        self.cn_scale = nn.Parameter(torch.tensor(50.0))

    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)
        
        for layer in self.mp_layers:
            x = layer(x, data.edge_index, edge_attr)
        
        # Global pooling
        graph_features = global_mean_pool(x, data.batch)
        
        # Predict CN from graph
        cn_pred = self.cn_head(graph_features) * self.cn_scale
        interaction_pred = self.interaction_head(graph_features)
        
        return cn_pred.squeeze(1), interaction_pred.squeeze(1)


class MixtureGNN(nn.Module):
    """Encode mixtures using per-component GNNs and linear blending."""
    def __init__(self, hidden_dim=64, num_layers=5, max_components=12):
        super().__init__()
        self.mol_gnn = MolecularGNN(hidden_dim, num_layers)
        self.max_components = max_components
        self.hidden_dim = hidden_dim
        
        # Linear blending weights
        self.component_weights = nn.Linear(max_components, 1, bias=False)

    def forward(self, component_graphs, mole_fractions):
        """
        Args:
            component_graphs: list of Data objects (one per component)
            mole_fractions: (batch_size, max_components) normalized volumes
        
        Returns:
            mixture_cn: (batch_size,) predicted cetane numbers
        """
        batch_size = mole_fractions.shape[0]
        component_cns = []
        
        for i in range(self.max_components):
            # Get graphs for this component across batch
            batch_data = [component_graphs[b][i] for b in range(batch_size)]
            
            # Skip if all None
            if all(g is None for g in batch_data):
                component_cns.append(torch.zeros(batch_size, device=mole_fractions.device))
                continue
            
            # Create batch
            valid_batch = [g for g in batch_data if g is not None]
            if len(valid_batch) == 0:
                component_cns.append(torch.zeros(batch_size, device=mole_fractions.device))
                continue
            
            batch = Batch.from_data_list(valid_batch)
            batch = batch.to(mole_fractions.device)
            
            cn_pred, _ = self.mol_gnn(batch)
            
            # Reconstruct full batch with zeros for missing components
            full_cn = torch.zeros(batch_size, device=mole_fractions.device)
            valid_idx = 0
            for b in range(batch_size):
                if batch_data[b] is not None:
                    full_cn[b] = cn_pred[valid_idx]
                    valid_idx += 1
            
            component_cns.append(full_cn)
        
        # Stack: (batch_size, max_components)
        component_cns = torch.stack(component_cns, dim=1)
        
        # Linear mixture blending
        mixture_cn = (component_cns * mole_fractions).sum(dim=1)
        
        return mixture_cn
