import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import GINConv, GINEConv
from torch_geometric.utils import degree
from torch_scatter import scatter
from typing import Dict, List, Optional

from models.layers import MLP, AtomEncoder, BondEncoder, FragEncoder, InterMessage


class FragGNN(torch.nn.Module):
    """
    Hierarchical Graph Neural Network for molecular fragments.

    This model implements a hierarchical message passing scheme between atoms and fragments
    in molecular graphs. It can optionally include graph-level representations and learned
    edge representations.

    Note: This class is deprecated. Use FragGNNSmall instead.

    Attributes:
        num_layers (int): Number of message passing layers
        hidden_channels (int): Dimension of hidden representations
        hidden_channels_substructure (int): Dimension of fragment representations
        dropout (float): Dropout probability
        inter_message_passing (bool): Whether to enable message passing between atoms and fragments
        higher_message_passing (bool): Whether to enable higher-level message passing
        low_high_edges (bool): Whether to use low-high edge features
        encoding_size_scaling (bool): Whether to scale encodings by size
        rbf (int): Number of radial basis functions
        degree_scaling (bool): Whether to scale features by node degree
        fragment_specific (bool): Whether to use fragment-specific message passing
        reduction (str): Reduction method for message aggregation
        frag_reduction (str): Reduction method for fragment message aggregation
        concat (bool): Whether to concatenate features
        graph_rep (bool): Whether to use graph-level representations
        graph_rep_node (bool): Whether to use node-level graph representations
        learned_edge_rep (bool): Whether to use learned edge representations
        higher_level_edge_features (bool): Whether to use higher-level edge features
        out_channels (int): Output dimension
        no_frag_info (bool): Whether to exclude fragment information
    """

    def __init__(
        self,
        in_channels: int,
        in_channels_substructure: int,
        in_channels_edge: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        linear_atom_encoder: bool = False,
        encoding_size_scaling: bool = False,
        rbf: int = 0,
        atom_feature_params: Dict = {},
        edge_feature_params: Dict = {},
        degree_scaling: bool = False,
        additional_atom_features: List = [],
        inter_message_passing: bool = True,
        higher_message_passing: bool = False,
        no_frag_info: bool = False,
        low_high_edges: bool = False,
        fragment_specific: bool = False,
        reduction: str = "mean",
        frag_reduction: Optional[str] = None,
        concat: bool = False,
        graph_rep: bool = False,
        learned_edge_rep: bool = False,
        higher_level_edge_features: bool = False,
        graph_rep_node: bool = False,
        inter_message_params: Dict = {},
        hidden_channels_substructure: Optional[int] = None,
        num_layers_out: int = 2,
    ) -> None:
        """
        Initialize the FragGNN model.

        Args:
            in_channels (int): Input dimension for atom features
            in_channels_substructure (int): Input dimension for fragment features
            in_channels_edge (int): Input dimension for edge features
            hidden_channels (int): Hidden dimension for atom representations
            out_channels (int): Output dimension
            num_layers (int): Number of message passing layers
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            linear_atom_encoder (bool, optional): Whether to use linear atom encoder. Defaults to False.
            encoding_size_scaling (bool, optional): Whether to scale encodings by size. Defaults to False.
            rbf (int, optional): Number of radial basis functions. Defaults to 0.
            atom_feature_params (Dict, optional): Parameters for atom feature encoding. Defaults to {}.
            edge_feature_params (Dict, optional): Parameters for edge feature encoding. Defaults to {}.
            degree_scaling (bool, optional): Whether to scale features by node degree. Defaults to False.
            additional_atom_features (List, optional): Additional atom features to encode. Defaults to [].
            inter_message_passing (bool, optional): Whether to enable message passing between atoms and fragments. Defaults to True.
            higher_message_passing (bool, optional): Whether to enable higher-level message passing. Defaults to False.
            no_frag_info (bool, optional): Whether to exclude fragment information. Defaults to False.
            low_high_edges (bool, optional): Whether to use low-high edge features. Defaults to False.
            fragment_specific (bool, optional): Whether to use fragment-specific message passing. Defaults to False.
            reduction (str, optional): Reduction method for message aggregation. Defaults to "mean".
            frag_reduction (Optional[str], optional): Reduction method for fragment message aggregation. Defaults to None.
            concat (bool, optional): Whether to concatenate features. Defaults to False.
            graph_rep (bool, optional): Whether to use graph-level representations. Defaults to False.
            learned_edge_rep (bool, optional): Whether to use learned edge representations. Defaults to False.
            higher_level_edge_features (bool, optional): Whether to use higher-level edge features. Defaults to False.
            graph_rep_node (bool, optional): Whether to use node-level graph representations. Defaults to False.
            inter_message_params (Dict, optional): Parameters for inter-message passing. Defaults to {}.
            hidden_channels_substructure (Optional[int], optional): Hidden dimension for fragment representations. Defaults to None.
            num_layers_out (int, optional): Number of output layers. Defaults to 2.
        """
        super(FragGNN, self).__init__()

        # Store configuration
        self.num_layers = num_layers
        self.hidden_channels_substructure = (
            hidden_channels_substructure if hidden_channels_substructure else hidden_channels
        )
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.inter_message_passing = inter_message_passing
        self.higher_message_passing = higher_message_passing
        self.low_high_edges = low_high_edges
        self.encoding_size_scaling = encoding_size_scaling
        self.rbf = rbf
        self.degree_scaling = degree_scaling
        self.fragment_specific = fragment_specific
        self.reduction = reduction
        self.frag_reduction = frag_reduction if frag_reduction else reduction
        self.concat = concat
        self.graph_rep = graph_rep
        self.graph_rep_node = graph_rep_node
        self.learned_edge_rep = learned_edge_rep
        self.higher_level_edge_features = higher_level_edge_features
        self.out_channels = out_channels
        self.no_frag_info = no_frag_info

        # Initialize encoders
        self.atom_encoder = (
            Linear(in_channels, hidden_channels)
            if linear_atom_encoder
            else AtomEncoder(hidden_channels, degree_scaling, additional_atom_features, **atom_feature_params)
        )

        self.clique_encoder = FragEncoder(
            in_channels_substructure, self.hidden_channels_substructure, encoding_size_scaling, rbf
        )

        # Initialize edge representation modules
        if not self.learned_edge_rep:
            self.bond_encoders = ModuleList()
        else:
            self.bond_encoder = BondEncoder(hidden_channels, **edge_feature_params)
            self.atom2bond = ModuleList()
            self.bond_batch_norms = ModuleList()
            self.bond_convs = ModuleList()

        # Initialize graph representation modules
        if self.graph_rep or self.graph_rep_node:
            hidden_channels_graph = hidden_channels
            self.graph_encoder = Embedding(1, hidden_channels_graph)

        if self.low_high_edges:
            self.bond_encoders_low_high = ModuleList()

        # Initialize atom-level modules
        self.atom_convs = ModuleList()
        self.atom_batch_norms = ModuleList()

        if self.graph_rep_node:
            self.atom2graph = ModuleList()
            self.graph2atom = ModuleList()

        if self.graph_rep or self.graph_rep_node:
            self.graph_batch_norms = ModuleList()
            self.graph_conv = ModuleList()

        # Build message passing layers
        for _ in range(num_layers):
            # Initialize bond encoders
            if not self.learned_edge_rep:
                self.bond_encoders.append(BondEncoder(hidden_channels, **edge_feature_params))
            if self.low_high_edges:
                self.bond_encoders_low_high.append(
                    BondEncoder(self.hidden_channels_substructure, **edge_feature_params)
                )

            # Initialize atom convolution layers
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.atom_convs.append(GINEConv(nn, train_eps=True, edge_dim=hidden_channels))
            self.atom_batch_norms.append(BatchNorm1d(hidden_channels))

            # Initialize graph representation layers
            if self.graph_rep_node:
                self.atom2graph.append(InterMessage(hidden_channels, hidden_channels_graph, **inter_message_params))
                self.graph2atom.append(Linear(hidden_channels_graph, hidden_channels))
            if self.graph_rep or self.graph_rep_node:
                self.graph_batch_norms.append(BatchNorm1d(hidden_channels_graph))
                self.graph_conv.append(Linear(hidden_channels_graph, hidden_channels_graph))

            # Initialize edge representation layers
            if self.learned_edge_rep:
                self.atom2bond.append(InterMessage(hidden_channels, hidden_channels, **inter_message_params))
                self.bond_batch_norms.append(BatchNorm1d(hidden_channels))
                self.bond_convs.append(Linear(hidden_channels, hidden_channels))

        # Initialize fragment-level modules
        if self.inter_message_passing:
            self.frag_convs = ModuleList()
            self.frag_batch_norms = ModuleList()
            if self.graph_rep:
                self.fragment2graph = ModuleList()
                self.graph2fragment = ModuleList()
            if self.concat:
                self.concat_lins = ModuleList()

            for _ in range(num_layers):
                # Initialize fragment convolution layers
                nn = Sequential(
                    Linear(self.hidden_channels_substructure, 2 * self.hidden_channels_substructure),
                    BatchNorm1d(2 * self.hidden_channels_substructure),
                    ReLU(),
                    Linear(2 * self.hidden_channels_substructure, self.hidden_channels_substructure),
                )
                if self.higher_level_edge_features:
                    self.frag_convs.append(GINEConv(nn, train_eps=True, edge_dim=self.hidden_channels))
                else:
                    self.frag_convs.append(GINConv(nn, train_eps=True))
                self.frag_batch_norms.append(BatchNorm1d(self.hidden_channels_substructure))

                # Initialize concatenation layers
                if self.concat:
                    self.concat_lins.append(Linear(2 * hidden_channels, hidden_channels))

                # Initialize graph-fragment interaction layers
                if self.graph_rep:
                    self.fragment2graph.append(
                        InterMessage(self.hidden_channels_substructure, hidden_channels_graph, **inter_message_params)
                    )
                    self.graph2fragment.append(Linear(hidden_channels_graph, self.hidden_channels_substructure))

            # Initialize atom-fragment interaction layers
            self.atom2frag = ModuleList()
            self.frag2atom = ModuleList()

            for _ in range(num_layers):
                if not self.fragment_specific:
                    self.atom2frag.append(
                        InterMessage(hidden_channels, self.hidden_channels_substructure, **inter_message_params)
                    )
                    self.frag2atom.append(
                        InterMessage(self.hidden_channels_substructure, hidden_channels, **inter_message_params)
                    )
                else:
                    self.atom2frag.append(
                        ModuleList(
                            [
                                InterMessage(hidden_channels, self.hidden_channels_substructure, **inter_message_params)
                                for _ in range(3)
                            ]
                        )
                    )
                    self.frag2atom.append(
                        ModuleList(
                            [
                                InterMessage(self.hidden_channels_substructure, hidden_channels, **inter_message_params)
                                for _ in range(3)
                            ]
                        )
                    )

        # Initialize output layers
        self.frag_out = MLP(self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        self.atom_out = MLP(self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        if self.learned_edge_rep:
            self.edge_out = MLP(self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        self.out = MLP(
            self.hidden_channels,
            self.out_channels,
            num_layers=num_layers_out,
            batch_norm=False,
            last_relu=False,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            data (torch.Tensor): Input graph data containing:
                - x: Atom features
                - edge_index: Edge connectivity
                - edge_attr: Edge features
                - batch: Batch assignments
                - fragments: Fragment features
                - fragment_types: Fragment type information
                - fragments_edge_index: Fragment connectivity

        Returns:
            torch.Tensor: Output predictions with shape (batch_size, out_channels)
        """
        batch_size = torch.max(data.batch) + 1

        # Encode atom features
        if self.degree_scaling:
            degrees = degree(data.edge_index[0], dtype=torch.float, num_nodes=data.x.size(0))
            atom_features = self.atom_encoder(data, degrees)
        else:
            atom_features = self.atom_encoder(data)

        # Encode fragment features
        if self.encoding_size_scaling:
            fragment_features = self.clique_encoder(data.fragment_types)
        else:
            fragment_features = self.clique_encoder(data.fragments)

        # Add fragment information to atom features if needed
        if not self.inter_message_passing and not self.no_frag_info:
            row, col = data.fragments_edge_index
            atom_features = atom_features + scatter(
                fragment_features[col], row, dim=0, dim_size=atom_features.size(0), reduce=self.reduction
            )

        # Initialize graph-level features if needed
        if self.graph_rep:
            graph_features = torch.zeros(batch_size, dtype=torch.int, device=atom_features.device)
            graph_features = self.graph_encoder(graph_features)

        # Initialize edge features if using learned edge representations
        if self.learned_edge_rep:
            edge_features = self.bond_encoder(data.edge_attr)

        # Message passing layers
        for layer_idx in range(self.num_layers):
            # Process atom-level messages
            if not self.learned_edge_rep:
                edge_features = self.bond_encoders[layer_idx](data.edge_attr)
            atom_features = self.atom_convs[layer_idx](atom_features, data.edge_index, edge_features)
            atom_features = self.atom_batch_norms[layer_idx](atom_features)
            atom_features = F.relu(atom_features)
            atom_features = F.dropout(atom_features, self.dropout, training=self.training)

            # Update graph-level features if using node-level graph representations
            if self.graph_rep_node:
                graph_features = graph_features + self.atom2graph[layer_idx](
                    atom_features, data.batch, dim_size=batch_size
                )

            # Update edge features if using learned edge representations
            if self.learned_edge_rep:
                row_edge, col_edge = data.edge_index
                edge_features = edge_features + self.atom2bond[layer_idx](
                    torch.concat([atom_features[row_edge], atom_features[col_edge]], dim=0),
                    torch.concat(
                        [torch.arange(row_edge.size(0), dtype=torch.int64, device=row_edge.device) for _ in range(2)],
                        dim=0,
                    ),
                    dim_size=row_edge.size(0),
                )
                edge_features = self.bond_convs[layer_idx](edge_features)
                edge_features = self.bond_batch_norms[layer_idx](edge_features)
                edge_features = F.relu(edge_features)
                edge_features = F.dropout(edge_features, self.dropout, training=self.training)

            # Process fragment-level messages if enabled
            if self.inter_message_passing:
                # Update fragment features
                fragment_features = self.frag_convs[layer_idx](fragment_features, data.fragments_edge_index)
                fragment_features = self.frag_batch_norms[layer_idx](fragment_features)
                fragment_features = F.relu(fragment_features)
                fragment_features = F.dropout(fragment_features, self.dropout, training=self.training)

                # Update graph-level features if using graph representations
                if self.graph_rep:
                    graph_features = graph_features + self.fragment2graph[layer_idx](
                        fragment_features, data.fragments_batch, dim_size=batch_size
                    )
                    graph_features = self.graph_batch_norms[layer_idx](graph_features)
                    graph_features = F.relu(graph_features)
                    graph_features = F.dropout(graph_features, self.dropout, training=self.training)
                    fragment_features = fragment_features + self.graph2fragment[layer_idx](graph_features)

                # Process atom-fragment interactions
                if not self.fragment_specific:
                    # Update atom features with fragment information
                    atom_features = atom_features + self.frag2atom[layer_idx](
                        fragment_features, data.fragments_edge_index[0], dim_size=atom_features.size(0)
                    )
                    # Update fragment features with atom information
                    fragment_features = fragment_features + self.atom2frag[layer_idx](
                        atom_features, data.fragments_edge_index[1], dim_size=fragment_features.size(0)
                    )
                else:
                    # Process fragment-specific interactions
                    for frag_type in range(3):
                        mask = data.fragment_types == frag_type
                        if mask.any():
                            atom_features = atom_features + self.frag2atom[layer_idx][frag_type](
                                fragment_features[mask],
                                data.fragments_edge_index[0][mask],
                                dim_size=atom_features.size(0),
                            )
                            fragment_features[mask] = fragment_features[mask] + self.atom2frag[layer_idx][frag_type](
                                atom_features,
                                data.fragments_edge_index[1][mask],
                                dim_size=fragment_features.size(0),
                            )

        # Process final representations
        fragment_features = self.frag_out(fragment_features)
        atom_features = self.atom_out(atom_features)
        if self.learned_edge_rep:
            edge_features = self.edge_out(edge_features)

        # Aggregate features for final prediction
        if self.graph_rep:
            return self.out(graph_features)
        elif self.graph_rep_node:
            return self.out(graph_features)
        else:
            return self.out(atom_features)


class FragGNNSmall(torch.nn.Module):
    """
    A simplified version of the FragGNN model for molecular fragment analysis.

    This model implements a streamlined hierarchical message passing scheme between atoms
    and fragments in molecular graphs, with fewer configuration options than the full
    FragGNN model.

    Attributes:
        num_layers (int): Number of message passing layers
        hidden_channels (int): Dimension of hidden representations
        hidden_channels_substructure (int): Dimension of fragment representations
        dropout (float): Dropout probability
        inter_message_passing (bool): Whether to enable message passing between atoms and fragments
        higher_message_passing (bool): Whether to enable higher-level message passing
        no_frag_info (bool): Whether to exclude fragment information
        reduction (str): Reduction method for message aggregation
        frag_reduction (str): Reduction method for fragment message aggregation
        learned_edge_rep (bool): Whether to use learned edge representations
        out_channels (int): Output dimension
    """

    def __init__(
        self,
        in_channels: int,
        in_channels_substructure: int,
        in_channels_edge: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        ordinal_encoding: bool = False,
        atom_feature_params: Dict = {},
        edge_feature_params: Dict = {},
        inter_message_passing: bool = True,
        higher_message_passing: bool = False,
        no_frag_info: bool = False,
        reduction: str = "mean",
        frag_reduction: Optional[str] = None,
        learned_edge_rep: bool = False,
        inter_message_params: Dict = {},
        hidden_channels_substructure: Optional[int] = None,
        num_layers_out: int = 2,
    ) -> None:
        """
        Initialize the FragGNNSmall model.

        Args:
            in_channels (int): Input dimension for atom features
            in_channels_substructure (int): Input dimension for fragment features
            in_channels_edge (int): Input dimension for edge features
            hidden_channels (int): Hidden dimension for atom representations
            out_channels (int): Output dimension
            num_layers (int): Number of message passing layers
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            ordinal_encoding (bool, optional): Whether to use ordinal encoding for fragments. Defaults to False.
            atom_feature_params (Dict, optional): Parameters for atom feature encoding. Defaults to {}.
            edge_feature_params (Dict, optional): Parameters for edge feature encoding. Defaults to {}.
            inter_message_passing (bool, optional): Whether to enable message passing between atoms and fragments. Defaults to True.
            higher_message_passing (bool, optional): Whether to enable higher-level message passing. Defaults to False.
            no_frag_info (bool, optional): Whether to exclude fragment information. Defaults to False.
            reduction (str, optional): Reduction method for message aggregation. Defaults to "mean".
            frag_reduction (Optional[str], optional): Reduction method for fragment message aggregation. Defaults to None.
            learned_edge_rep (bool, optional): Whether to use learned edge representations. Defaults to False.
            inter_message_params (Dict, optional): Parameters for inter-message passing. Defaults to {}.
            hidden_channels_substructure (Optional[int], optional): Hidden dimension for fragment representations. Defaults to None.
            num_layers_out (int, optional): Number of output layers. Defaults to 2.
        """
        super(FragGNNSmall, self).__init__()

        # Store configuration
        self.num_layers = num_layers
        self.hidden_channels_substructure = (
            hidden_channels_substructure if hidden_channels_substructure else hidden_channels
        )
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.inter_message_passing = inter_message_passing
        self.higher_message_passing = higher_message_passing
        self.reduction = reduction
        self.frag_reduction = frag_reduction if frag_reduction else reduction
        self.learned_edge_rep = learned_edge_rep
        self.out_channels = out_channels
        self.no_frag_info = no_frag_info

        # Initialize encoders
        self.atom_encoder = AtomEncoder(hidden_channels, **atom_feature_params)
        self.clique_encoder = FragEncoder(in_channels_substructure, self.hidden_channels_substructure, ordinal_encoding)

        # Initialize edge representation modules
        if not self.learned_edge_rep:
            self.bond_encoders = ModuleList()
        else:
            self.bond_encoder = BondEncoder(hidden_channels, **edge_feature_params)
            self.atom2bond = ModuleList()
            self.bond_batch_norms = ModuleList()
            self.bond_convs = ModuleList()

        # Initialize atom-level modules
        self.atom_convs = ModuleList()
        self.atom_batch_norms = ModuleList()

        # Build message passing layers
        for _ in range(num_layers):
            # Initialize bond encoders
            if not self.learned_edge_rep:
                self.bond_encoders.append(BondEncoder(hidden_channels, **edge_feature_params))

            # Initialize atom convolution layers
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.atom_convs.append(GINEConv(nn, train_eps=True, edge_dim=hidden_channels))
            self.atom_batch_norms.append(BatchNorm1d(hidden_channels))

            # Initialize edge representation layers
            if self.learned_edge_rep:
                self.atom2bond.append(InterMessage(hidden_channels, hidden_channels, **inter_message_params))
                self.bond_batch_norms.append(BatchNorm1d(hidden_channels))
                self.bond_convs.append(Linear(hidden_channels, hidden_channels))

        # Initialize fragment-level modules
        if self.inter_message_passing:
            self.frag_convs = ModuleList()
            self.frag_batch_norms = ModuleList()

            for _ in range(num_layers):
                # Initialize fragment convolution layers
                nn = Sequential(
                    Linear(self.hidden_channels_substructure, 2 * self.hidden_channels_substructure),
                    BatchNorm1d(2 * self.hidden_channels_substructure),
                    ReLU(),
                    Linear(2 * self.hidden_channels_substructure, self.hidden_channels_substructure),
                )
                self.frag_convs.append(GINConv(nn, train_eps=True))
                self.frag_batch_norms.append(BatchNorm1d(self.hidden_channels_substructure))

            # Initialize atom-fragment interaction layers
            self.atom2frag = ModuleList()
            self.frag2atom = ModuleList()

            for _ in range(num_layers):
                self.atom2frag.append(
                    InterMessage(hidden_channels, self.hidden_channels_substructure, **inter_message_params)
                )
                self.frag2atom.append(
                    InterMessage(self.hidden_channels_substructure, hidden_channels, **inter_message_params)
                )

        # Initialize output layers
        self.frag_out = MLP(self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        self.atom_out = MLP(self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        if self.learned_edge_rep:
            self.edge_out = MLP(self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        self.out = MLP(
            self.hidden_channels,
            self.out_channels,
            num_layers=num_layers_out,
            batch_norm=False,
            last_relu=False,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            data (torch.Tensor): Input graph data containing:
                - x: Atom features
                - edge_index: Edge connectivity
                - edge_attr: Edge features
                - batch: Batch assignments
                - fragments: Fragment features
                - fragments_edge_index: Fragment connectivity
                - fragments_batch: Fragment batch assignments

        Returns:
            torch.Tensor: Output predictions with shape (batch_size, out_channels)
        """
        batch_size = torch.max(data.batch) + 1

        # Encode atom features
        atom_features = self.atom_encoder(data)

        # Encode fragment features
        fragment_features = self.clique_encoder(data.fragments)

        # Add fragment information to atom features if needed
        if not self.inter_message_passing and not self.no_frag_info:
            row, col = data.fragments_edge_index
            atom_features = atom_features + scatter(
                fragment_features[col], row, dim=0, dim_size=atom_features.size(0), reduce=self.reduction
            )

        # Initialize edge features if using learned edge representations
        if self.learned_edge_rep:
            edge_features = self.bond_encoder(data.edge_attr)

        # Message passing layers
        for layer_idx in range(self.num_layers):
            # Process atom-level messages
            if not self.learned_edge_rep:
                edge_features = self.bond_encoders[layer_idx](data.edge_attr)
            atom_features = self.atom_convs[layer_idx](atom_features, data.edge_index, edge_features)
            atom_features = self.atom_batch_norms[layer_idx](atom_features)
            atom_features = F.relu(atom_features)
            atom_features = F.dropout(atom_features, self.dropout, training=self.training)

            # Update edge features if using learned edge representations
            if self.learned_edge_rep:
                row_edge, col_edge = data.edge_index
                edge_features = edge_features + self.atom2bond[layer_idx](
                    torch.concat([atom_features[row_edge], atom_features[col_edge]], dim=0),
                    torch.concat(
                        [torch.arange(row_edge.size(0), dtype=torch.int64, device=row_edge.device) for _ in range(2)],
                        dim=0,
                    ),
                    dim_size=row_edge.size(0),
                )
                edge_features = self.bond_convs[layer_idx](edge_features)
                edge_features = self.bond_batch_norms[layer_idx](edge_features)
                edge_features = F.relu(edge_features)
                edge_features = F.dropout(edge_features, self.dropout, training=self.training)

            # Process fragment-level messages if enabled
            if self.inter_message_passing:
                # Update fragment features
                fragment_features = self.frag_convs[layer_idx](fragment_features, data.fragments_edge_index)
                fragment_features = self.frag_batch_norms[layer_idx](fragment_features)
                fragment_features = F.relu(fragment_features)
                fragment_features = F.dropout(fragment_features, self.dropout, training=self.training)

                # Process atom-fragment interactions
                row, col = data.fragments_edge_index

                # Update atom features with fragment information
                atom_features = atom_features + self.frag2atom[layer_idx](
                    fragment_features, row, dim_size=atom_features.size(0)
                )

                # Update fragment features with atom information
                fragment_features = fragment_features + self.atom2frag[layer_idx](
                    atom_features, col, dim_size=fragment_features.size(0)
                )

        # Process final representations
        fragment_features = self.frag_out(fragment_features)
        atom_features = self.atom_out(atom_features)
        if self.learned_edge_rep:
            edge_features = self.edge_out(edge_features)

        # Aggregate features for final prediction
        atom_features = scatter(atom_features, data.batch, dim=0, reduce=self.reduction)
        atom_features = F.dropout(atom_features, self.dropout, training=self.training)

        if self.inter_message_passing:
            fragment_features = scatter(
                fragment_features,
                data.fragments_batch,
                dim=0,
                dim_size=atom_features.size(0),
                reduce=self.frag_reduction,
            )
            fragment_features = F.dropout(fragment_features, self.dropout, training=self.training)
            atom_features = atom_features + fragment_features

        if self.learned_edge_rep:
            edge_batch = data.batch[data.edge_index[0]]
            edge_features = scatter(edge_features, edge_batch, dim=0, dim_size=batch_size, reduce=self.reduction)
            edge_features = F.dropout(edge_features, self.dropout, training=self.training)
            atom_features = atom_features + edge_features

        return self.out(atom_features)
