import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Embedding, Linear, ReLU, Sequential
from torch_scatter import scatter
from typing import List, Optional


class AtomEncoder(torch.nn.Module):
    """
    Encodes atom features into a continuous vector representation.

    This module converts discrete atom features (like atom type, formal charge, etc.)
    into continuous embeddings. It can optionally scale features based on node degree.

    Attributes:
        degree_scaling (bool): Whether to scale features by node degree
        hidden_channels (int): Dimension of the output embeddings
        embeddings (ModuleList): List of embedding layers for each atom feature
    """

    def __init__(
        self, hidden_channels: int, degree_scaling: bool = False, num_atom_types: int = 100, num_atom_features: int = 9
    ) -> None:
        """
        Initialize the AtomEncoder.

        Args:
            hidden_channels (int): Dimension of the output embeddings
            degree_scaling (bool, optional): Whether to scale features by node degree. Defaults to False.
            num_atom_types (int, optional): Maximum number of unique atom types. Defaults to 100.
            num_atom_features (int, optional): Number of atom features to encode. Defaults to 9.
        """
        super(AtomEncoder, self).__init__()
        self.degree_scaling = degree_scaling
        self.hidden_channels = hidden_channels

        # Create an embedding layer for each atom feature
        self.embeddings = torch.nn.ModuleList(
            [Embedding(num_atom_types, hidden_channels) for _ in range(num_atom_features)]
        )

    def forward(self, graph: torch.Tensor, degree_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode atom features into continuous embeddings.

        Args:
            graph (torch.Tensor): Input graph containing atom features
            degree_info (Optional[torch.Tensor]): Node degree information for scaling. Required if degree_scaling is True.

        Returns:
            torch.Tensor: Encoded atom features with shape (num_atoms, hidden_channels)
        """
        # Ensure input has correct shape
        atom_features = graph.x
        if atom_features.dim() == 1:
            atom_features = atom_features.unsqueeze(1)

        # Sum embeddings from all features
        encoded_features = torch.zeros(atom_features.size(0), self.hidden_channels, device=atom_features.device)
        for feature_idx in range(atom_features.size(1)):
            encoded_features += self.embeddings[feature_idx](atom_features[:, feature_idx])

        # Apply degree scaling if enabled
        if self.degree_scaling and degree_info is not None:
            half_channels = self.hidden_channels // 2
            encoded_features[:, :half_channels] = (
                torch.unsqueeze(degree_info, dim=1) * encoded_features[:, :half_channels]
            )

        return encoded_features


class BondEncoder(torch.nn.Module):
    """
    Encodes bond features into a continuous vector representation.

    This module converts discrete bond features (like bond type, stereo, etc.)
    into continuous embeddings.

    Attributes:
        embeddings (ModuleList): List of embedding layers for each bond feature
    """

    def __init__(self, hidden_channels: int, num_bond_types: int = 6, num_bond_features: int = 3) -> None:
        """
        Initialize the BondEncoder.

        Args:
            hidden_channels (int): Dimension of the output embeddings
            num_bond_types (int, optional): Maximum number of unique bond types. Defaults to 6.
            num_bond_features (int, optional): Number of bond features to encode. Defaults to 3.
        """
        super(BondEncoder, self).__init__()

        # Create an embedding layer for each bond feature
        self.embeddings = torch.nn.ModuleList(
            [Embedding(num_bond_types, hidden_channels) for _ in range(num_bond_features)]
        )

    def forward(self, bond_attr: torch.Tensor) -> torch.Tensor:
        """
        Encode bond features into continuous embeddings.

        Args:
            bond_attr (torch.Tensor): Input bond attributes of shape (num_bonds, num_bond_features)
                where each entry < num_bond_types

        Returns:
            torch.Tensor: Encoded bond features with shape (num_bonds, hidden_channels)
        """
        # Ensure input has correct shape
        if bond_attr.dim() == 1:
            bond_attr = bond_attr.unsqueeze(1)

        # Sum embeddings from all features
        encoded_features = torch.zeros(bond_attr.size(0), self.embeddings[0].embedding_dim, device=bond_attr.device)
        for feature_idx in range(bond_attr.size(1)):
            encoded_features += self.embeddings[feature_idx](bond_attr[:, feature_idx])

        return encoded_features


class FragEncoder(torch.nn.Module):
    """
    Encodes molecular fragment features into a continuous vector representation.

    This module can handle both ordinal and non-ordinal encoding of fragment features.
    For ordinal encoding, it considers both fragment class and size.

    Attributes:
        ordinal_encoding (bool): Whether to use ordinal encoding
        hidden_channels (int): Dimension of the output embeddings
        embedding (Embedding): Embedding layer for fragment features
    """

    def __init__(self, in_channels_substructure: int, hidden_channels: int, ordinal_encoding: bool) -> None:
        """
        Initialize the FragEncoder.

        Args:
            in_channels_substructure (int): Number of different substructure types (for non-ordinal encoding)
            hidden_channels (int): Dimension of the output embeddings
            ordinal_encoding (bool): Whether to use ordinal encoding
        """
        super(FragEncoder, self).__init__()
        self.ordinal_encoding = ordinal_encoding
        self.hidden_channels = hidden_channels

        if not self.ordinal_encoding:
            self.embedding = Embedding(in_channels_substructure, hidden_channels)
        else:
            # Embed paths, junction, ring with ordinal encoding
            self.embedding = Embedding(4, hidden_channels)

    def forward(self, frag_attr: torch.Tensor) -> torch.Tensor:
        """
        Encode fragment features into continuous embeddings.

        Args:
            frag_attr (torch.Tensor): Input fragment attributes.
                If not ordinal_encoding: one-hot encoded, shape (num_fragments, in_channels_substructure)
                If ordinal_encoding: shape (num_fragments, 2) where first column is fragment class
                and second column is size

        Returns:
            torch.Tensor: Encoded fragment features with shape (num_fragments, hidden_channels)
        """
        if not self.ordinal_encoding:
            # Convert one-hot encoding to indices
            fragment_indices = torch.argmax(frag_attr, dim=1)
            return self.embedding(fragment_indices)
        else:
            assert frag_attr.size(1) == 2, "Ordinal encoding requires 2 features: class and size"

            # Get base embeddings from fragment class
            encoded_features = self.embedding(frag_attr[:, 0])

            # Scale first half of features by fragment size
            half_channels = self.hidden_channels // 2
            encoded_features[:, :half_channels] = (
                torch.unsqueeze(frag_attr[:, 1], dim=1) * encoded_features[:, :half_channels]
            )

            return encoded_features


class InterMessage(torch.nn.Module):
    """
    Handles message passing between different levels of the molecular hierarchy.

    This module can transform messages either before or after scattering,
    and supports different reduction operations.

    Attributes:
        transform_scatter (bool): Whether to transform before scattering
        reduction (str): Reduction method for scatter operation
        transform (Sequential): MLP for transforming messages
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        transform_scatter: bool = False,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize the InterMessage module.

        Args:
            in_channels (int): Input dimension
            out_channels (int): Output dimension
            num_layers (int, optional): Number of MLP layers. Defaults to 1.
            transform_scatter (bool, optional): Whether to transform before scattering. Defaults to False.
            reduction (str, optional): Reduction method for scatter. Defaults to "mean".
        """
        super(InterMessage, self).__init__()

        self.transform_scatter = transform_scatter
        self.reduction = reduction

        # Build MLP layers
        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else out_channels
            layers.extend([Linear(in_dim, out_channels), ReLU()])

        self.transform = Sequential(*layers)

    def forward(self, from_tensor: torch.Tensor, to_index: torch.Tensor, dim_size: int) -> torch.Tensor:
        """
        Perform message passing between hierarchy levels.

        Args:
            from_tensor (torch.Tensor): Source tensor of shape (num_nodes, in_channels)
            to_index (torch.Tensor): Target indices for scattering
            dim_size (int): Size of the target dimension

        Returns:
            torch.Tensor: Transformed and scattered messages with shape (dim_size, out_channels)
        """
        if self.transform_scatter:
            # Transform first, then scatter
            transformed = self.transform(from_tensor)
            return scatter(transformed, to_index, dim=0, dim_size=dim_size, reduce=self.reduction)
        else:
            # Scatter first, then transform
            scattered = scatter(from_tensor, to_index, dim=0, dim_size=dim_size, reduce=self.reduction)
            return self.transform(scattered)


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron with optional batch normalization.

    This module implements a fully connected neural network with configurable
    number of layers, batch normalization, and activation functions.

    Attributes:
        nn (Sequential): The neural network layers
    """

    def __init__(
        self, in_channels: int, out_channels: int, num_layers: int, batch_norm: bool = True, last_relu: bool = True
    ) -> None:
        """
        Initialize the MLP.

        Args:
            in_channels (int): Input dimension
            out_channels (int): Output dimension
            num_layers (int): Number of hidden layers
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.
            last_relu (bool, optional): Whether to apply ReLU after last layer. Defaults to True.
        """
        super(MLP, self).__init__()

        # Calculate hidden layer sizes
        hidden_sizes = np.linspace(start=in_channels, stop=out_channels, num=num_layers + 1, endpoint=True, dtype=int)

        # Build layers
        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_dim = hidden_sizes[i]
            out_dim = hidden_sizes[i + 1]

            layers.append(Linear(in_dim, out_dim))

            if batch_norm:
                layers.append(BatchNorm1d(out_dim))

            if i != num_layers - 1 or last_relu:
                layers.append(ReLU())

        self.nn = Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels)
        """
        return self.nn(x)


class MLPReadout(nn.Module):
    """
    MLP-based readout module for graph-level predictions.

    This module implements a hierarchical MLP that gradually reduces the
    feature dimension through a series of linear layers.

    Attributes:
        linear_layers (ModuleList): List of linear layers
        num_hidden_layers (int): Number of hidden layers
    """

    def __init__(self, input_dim: int, output_dim: int, num_hidden_layers: int = 2) -> None:
        """
        Initialize the MLPReadout module.

        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 2.
        """
        super().__init__()

        # Create layers with gradually reducing dimensions
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(
                    input_dim // 2**layer,
                    input_dim // 2 ** (layer + 1) if layer < num_hidden_layers else output_dim,
                    bias=True,
                )
                for layer in range(num_hidden_layers + 1)
            ]
        )
        self.num_hidden_layers = num_hidden_layers

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the readout MLP.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        x = input_tensor
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        return x
