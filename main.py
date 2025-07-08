import torch
from torch.optim import Adam
from dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from net import Net
from utils import plot_voronoi_from_logits


def cross_entropy_loss(gt: torch.Tensor, pred: torch.Tensor, eps=1e-12) -> torch.Tensor:
    """Calculate Cross Entropy Loss between two tensors"""
    batch_size = pred.shape[0]
    pred = torch.clamp(pred, min=eps, max=1.0 - eps)
    return -torch.sum(gt * torch.log(pred)) / batch_size


if __name__ == "__main__":
    dataset = CustomImageDataset("data/semantic_word_dataset.txt")
    dataloader = DataLoader(dataset=dataset, batch_size=16)
    net = Net(input_size=dataset.n_embeddings, n_classes=dataset.n_classes)
    optimizer = Adam(net.parameters())
    N_EPOCHS = 350

    for epoch in range(N_EPOCHS):
        batch_loss = []
        for data, target in dataloader:
            pred = net.forward(data)
            loss = cross_entropy_loss(target, pred)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss)

        if epoch % 25 == 0:
            loss = sum(batch_loss) / len(batch_loss)
            plot_voronoi_from_logits(net, epoch, loss)
            print(loss)
