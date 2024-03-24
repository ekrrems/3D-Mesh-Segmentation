import model
import numpy as np
import trimesh

def visualize_data(mesh, labels):
    """Visualize segmentation labels on a 3D mesh.

    Args:
        mesh (trimesh.base.Trimesh): The 3D mesh object.
        labels (torch.Tensor): Predicted segmentation labels for each vertex.

    Returns:
        trimesh.base.Trimesh: The 3D mesh object with vertex colors updated based on segmentation labels.

    Raises:
        ValueError: If the number of colors does not match the number of vertices in the mesh.

    """
    colors = []
    for label in labels:
        label = list(model.segmentation_colors.keys())[label.item()]
        color = model.segmentation_colors.get(label)
        colors.append(color.numpy())

    if len(colors) != len(mesh.vertices):
        raise ValueError("Number of colors does not match number of vertices")

    mesh.visual.vertex_colors = colors

    return mesh

def get_predictions(model_path ,data, device):
    """Get segmentation predictions for input data using a trained model.

    Args:
        model_path (str): Path to the trained model checkpoint.
        data (torch_geometric.data.Data): Input data.
        device (str): Device to run the inference on (e.g., "cpu", "cuda").

    Returns:
        tuple: A tuple containing the segmented mesh and predicted segmentation labels.

    """
    data = data.to(device)
    net = model.load_model(model.model_params, model_path, device)
    predictions = net(data)
    predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)

    segmented_mesh = trimesh.base.Trimesh(
        vertices=data.x.cpu().numpy(),
        faces=data.face.t().cpu().numpy(),
        process=False,
    )
    return segmented_mesh, predicted_seg_labels

# Visualize segmented 3D meshes
segmented_meshes = []
mesh_ids = range(0, 20, 3)
for idx, mesh_id in enumerate(mesh_ids):
    segmented_mesh = visualize_data(*get_predictions("mesh_segmentation_model", model.test_data[idx], "cpu"))
    segmented_mesh.vertices += [idx * 1.0, 0.0, 0.0]
    segmented_meshes.append(segmented_mesh)

scene = trimesh.scene.Scene(segmented_meshes)
scene.show()