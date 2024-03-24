import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.transforms import BaseTransform, Compose, FaceToEdge
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

import numpy as np
import os
import trimesh
from pathlib import Path
from functools import lru_cache

# Dictionary containing segmentation colors for different body parts
segmentation_colors = dict(
    head=torch.tensor([255, 0, 0], dtype=torch.int),                # Head (Red)
    torso=torch.tensor([0, 255, 0], dtype=torch.int),               # Torso (Green)
    left_arm=torch.tensor([0, 0, 255], dtype=torch.int),            # Left Arm (Blue)
    left_hand=torch.tensor([255, 255, 0], dtype=torch.int),         # Left Hand (Yellow)
    right_arm=torch.tensor([255, 0, 255], dtype=torch.int),         # Right Arm (Magenta)
    right_hand=torch.tensor([0, 255, 255], dtype=torch.int),        # Right Hand (Cyan)
    left_upper_leg=torch.tensor([255, 128, 0], dtype=torch.int),    # Left Upper Leg (Orange)
    left_lower_leg=torch.tensor([128, 0, 255], dtype=torch.int),    # Left Lower Leg (Purple)
    left_foot=torch.tensor([255, 0, 128], dtype=torch.int),         # Left Foot (Pink)
    right_upper_leg=torch.tensor([0, 255, 128], dtype=torch.int),   # Right Upper Leg (Light Green)
    right_lower_leg=torch.tensor([128, 255, 0], dtype=torch.int),   # Right Lower Leg (Light Blue)
    right_foot=torch.tensor([0, 128, 255], dtype=torch.int)         # Right Foot (Light Orange)
)

# Device specification
device = "cpu"

def load_mesh(mesh_path):
    """Loads a mesh from the given file path."""

    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = torch.from_numpy(mesh.vertices).to(torch.float)
    faces = torch.from_numpy(mesh.faces)
    faces = faces.T.to(torch.long).contiguous()
    return vertices, faces


def create_data(pre_transform=None):
    """Create data for segmentation task.

    Args:
        pre_transform (callable, optional): A function to be applied to the data before returning.

    Returns:
        list: A list of `torch_geometric.data.Data` objects containing mesh data and segmentation labels.

    """
    path_to_labels = os.path.join("dataset", "MPI-FAUST", "segmentations.npz")
    seg_labels = np.load(str(path_to_labels))["segmentation_labels"]
    seg_labels = torch.from_numpy(seg_labels).type(torch.int64)

    path_to_meshes = Path(os.path.join("dataset", "MPI-FAUST", "meshes"))
    mesh_filenames = path_to_meshes.glob("*.ply")

    data_list = []
    for mesh_filename in sorted(mesh_filenames):
        vertices, faces = load_mesh(mesh_filename)
        data = Data(x=vertices, face=faces)
        data.segmentation_labels = seg_labels
        if pre_transform is not None:
            data = pre_transform(data)
        data_list.append(data)

    return data_list


class NormalizeUnitSphere(BaseTransform):
    @staticmethod
    def _recenter(x):
        centroid = torch.mean(x)
        return x - centroid

    @staticmethod
    def _rescale_to_unit_length(x):
        max_dist = torch.max(torch.norm(x, dim=1))
        return x / max_dist
    
    def __call__(self, data: Data):
        if data.x is not None:
            data.x = self._rescale_to_unit_length(self._recenter(data.x))
        
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    

## helper functions
def pairwise(iterable):
    """Iterate over all pairs of consecutive items in a list.
    Notes
    -----
        [s0, s1, s2, s3, ...] -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    for i in range(len(iterable) - 1):
        yield (iterable[i], iterable[i + 1])


def get_conv_layers(channels: list, conv: MessagePassing, conv_params: dict):
    """Create a list of convolutional layers for a graph neural network.

    Parameters:
        channels (list): A list of integers representing the number of input channels
                         and output channels for each convolutional layer.
        conv (MessagePassing): The type of graph convolutional operation to use.
        conv_params (dict): Additional parameters to be passed to the convolutional layers.

    Returns:
        list: A list of convolutional layers initialized according to the provided configuration.

    Example:
        # Define a list of channels for each convolutional layer
        channels = [64, 64, 128, 128]

        # Initialize the graph convolutional layers
        conv_layers = get_conv_layers(channels, GCNConv, {'aggr': 'add'})

    """
    
    conv_layers = [
        conv(in_ch, out_ch, **conv_params) for in_ch, out_ch in pairwise(channels)
    ]
    return conv_layers


def get_mlp_layers(channels: list, activation, output_activation=nn.Identity):
    layers = []
    *intermediate_layer_definitions, final_layer_definition = pairwise(channels)

    for in_ch, out_ch in intermediate_layer_definitions:
        intermediate_layer = nn.Linear(in_ch, out_ch)
        layers += [intermediate_layer, activation()]

    layers += [nn.Linear(*final_layer_definition), output_activation()]
    return nn.Sequential(*layers)


## Feature-Steered Graph Convolution
class FeatureSteeredConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_heads: int,
            ensure_trans_invar: bool = True,
            bias: bool = True,
            with_self_loops: bool = True):
        super().__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.with_self_loops = with_self_loops

        self.linear = torch.nn.Linear(
            in_features=in_channels,
            out_features=out_channels * num_heads,
            bias=False,
        )

        self.u = torch.nn.Linear(
            in_features=in_channels,
            out_features=num_heads,
            bias=False,
        )

        self.c = torch.nn.Parameter(torch.Tensor(num_heads))

        if not ensure_trans_invar:
            self.v = torch.nn.Linear(
                in_features=in_channels,
                out_features=num_heads,
                bias=False,
            )
        else:
            self.register_parameter("v", None)
        
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialization of tuneable network parameters."""
        torch.nn.init.uniform_(self.linear.weight)
        torch.nn.init.uniform_(self.u.weight)
        torch.nn.init.normal_(self.c, mean=0.0, std=0.1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0.0, std=0.1)
        if self.v is not None:
            torch.nn.init.uniform_(self.v.weight)

    def forward(self, x, edge_index):
        if self.with_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=x.shape[0])

        out = self.propagate(edge_index, x=x)
        return out if self.bias is None else out + self.bias
    
    def _compute_attention_weights(self, x_i, x_j):
        if x_j.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected input features with {self.in_channels} channels."
                f" Instead received features with {x_j.shape[-1]} channels."
            )
        
        if self.v is None:
            attention_logits = self.u(x_i - x_j) + self.c
        else:
            attention_logits = self.u(x_i) + self.b(x_j) + self.c

        return F.softmax(attention_logits, dim=1)
    
    def message(self, x_i, x_j):
        attention_weights = self._compute_attention_weights(x_i, x_j)
        x_j = self.linear(x_j).view(-1, self.num_heads, self.out_channels)
        return (attention_weights.view(-1, self.num_heads, 1) * x_j).sum(dim=1)
    

### Garph Neural Ntwork
class GraphFeatureEncoder(nn.Module):
    def __init__(
        self,
        in_features,
        conv_channels,
        num_heads,
        apply_batch_norm: int = True,
        ensure_trans_invar: bool = True,
        bias: bool = True,
        with_self_loops: bool = True,
    ):
        super().__init__()

        conv_params = dict(
            num_heads=num_heads,
            ensure_trans_invar=ensure_trans_invar,
            bias=bias,
            with_self_loops=with_self_loops,
        )

        self.apply_batch_norm = apply_batch_norm

        *first_conv_channels, final_conv_channel = conv_channels

        conv_layers = get_conv_layers(
            channels=[in_features] + conv_channels,
            conv=FeatureSteeredConv,
            conv_params=conv_params,
        )

        self.conv_layers = nn.ModuleList(conv_layers)
    
        self.batch_layers = [None for _ in first_conv_channels]
        if apply_batch_norm:
            self.batch_layers = nn.ModuleList(
                [nn.BatchNorm1d(channel) for channel in first_conv_channels]
            )
        
    def forward(self, x, edge_index):
        *first_conv_layers, final_conv_layer = self.conv_layers
        for conv_layer, batch_layer in zip(first_conv_layers, self.batch_layers):
            x = conv_layer(x, edge_index)
            x = F.relu(x)
            if batch_layer is not None:
                x = batch_layer(x)
        return final_conv_layer(x, edge_index)
    

class MeshSeg(torch.nn.Module):
    """Mesh segmentation network."""
    def __init__(
        self,
        in_features,
        encoder_features,
        conv_channels,
        encoder_channels,
        decoder_channels,
        num_classes,
        num_heads,
        apply_batch_norm=True,
    ):
        super().__init__()
        self.input_encoder = get_mlp_layers(
            channels=[in_features] + encoder_channels,
            activation=nn.ReLU,
        )
        self.gnn = GraphFeatureEncoder(
            in_features=encoder_features,
            conv_channels=conv_channels,
            num_heads=num_heads,
            apply_batch_norm=apply_batch_norm,
        )
        *_, final_conv_channel = conv_channels

        self.final_projection = get_mlp_layers(
            [final_conv_channel] + decoder_channels + [num_classes],
            activation=nn.ReLU,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_encoder(x)
        x = self.gnn(x, edge_index)
        return self.final_projection(x)

def train(net, train_data, optimizer, loss_fn, device):
    """Train network on training dataset."""
    net.train()
    cumulative_loss = 0.0
    for data in train_data:
        data = data.to(device)
        optimizer.zero_grad()
        out = net(data)
        loss = loss_fn(out, data.segmentation_labels.squeeze())
        loss.backward()
        cumulative_loss += loss.item()
        optimizer.step()
    return cumulative_loss / len(train_data)


def accuracy(predictions, gt_seg_labels):
    """Calculate the accuracy of the model predictions"""
    predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)
    if predicted_seg_labels.shape != gt_seg_labels.shape:
        raise ValueError("Expected Shapes to be equivalent")
    correct_assignments = (predicted_seg_labels == gt_seg_labels).sum()
    num_assignemnts = predicted_seg_labels.shape[0]
    return float(correct_assignments / num_assignemnts)


def evaluate_performance(dataset, net, device):
    prediction_accuracies = []
    for data in dataset:
        data = data.to(device)
        predictions = net(data)
        prediction_accuracies.append(accuracy(predictions, data.segmentation_labels))
    return sum(prediction_accuracies) / len(prediction_accuracies)


@torch.no_grad()
def test(net, train_data, test_data, device):
    """Evaluate model's accuracy"""
    net.eval()
    train_acc = evaluate_performance(train_data, net, device)
    test_acc = evaluate_performance(test_data, net, device)
    return train_acc, test_acc

# Parameter for the model
model_params = dict(
    in_features=3,
    encoder_features=16,
    conv_channels=[32, 64, 128, 64],
    encoder_channels=[16],
    decoder_channels=[32],
    num_classes=12,
    num_heads=12,
    apply_batch_norm=True,
)

# Transform data prior training
pre_transform = Compose([FaceToEdge(remove_faces=False), NormalizeUnitSphere()])

# Create data
train_data = create_data(pre_transform=pre_transform)[:80]
test_data = create_data(pre_transform=pre_transform)[80:]
train_loader = DataLoader(train_data,  shuffle=True)
test_loader = DataLoader(test_data, shuffle=False)

def load_model(model_params, path_to_checkpoint, device):
    """Load a pre-trained model.

    Args:
        model_params (dict): Parameters to initialize the model.
        path_to_checkpoint (str): Path to the checkpoint file containing the model's state dictionary.
        device (torch.device): Device on which to load the model.

    Returns:
        torch.nn.Module: The loaded pre-trained model.

    Raises:
        ValueError: If the checkpoint cannot be loaded.

    """
    try:
        model = MeshSeg(**model_params)
        model.load_state_dict(
            torch.load(str(path_to_checkpoint)),
            strict=True,
        )
        model.to(device)
        return model
    except RuntimeError as err_msg:
        raise ValueError(
            f"Given checkpoint {str(path_to_checkpoint)} could"
            f" not be loaded. {err_msg}"
        )
