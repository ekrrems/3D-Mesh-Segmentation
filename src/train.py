import model
import torch
from tqdm import tqdm

def train_model(device):
    """Train the mesh segmentation model.

    Args:
        device (str): Device to train the model on (e.g., "cpu", "cuda").

    """
    lr = 0.001
    num_epochs = 70

    net = model.MeshSeg(**model.model_params).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    with tqdm(range(num_epochs), unit="Epoch") as tepochs:
        for epoch in tepochs:
            train_loss = model.train(net, model.train_loader, optimizer, loss_fn, device)
            train_acc, test_acc = model.test(net, model.train_loader, model.test_loader, device)
            print(f"train accuracy, train loss => {train_acc}, {train_loss}")
            print(f"Test accuracy => {test_acc}")
            
            tepochs.set_postfix(
                train_loss=train_loss,
                train_accuracy=100 * train_acc,
                test_accuracy=100 * test_acc,
            )

    torch.save(net.state_dict(), "mesh_segmentation_model")

if __name__ == "__main__":
    train_model('cpu')