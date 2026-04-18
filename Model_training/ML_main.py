import os
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from Model_training.get_dataset_from_npy import get_dataset_from_npy
from Model_training.NeuralNetwork import NeuralNetwork
from torch import nn

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (anchor, positive, negative) in enumerate(dataloader):
        anchor_tensor = torch.stack(anchor).float()
        positive_tensor = torch.stack(positive).float()
        negative_tensor = torch.stack(negative).float()
        anchor_tensor = anchor_tensor.view(1, -1)
        positive_tensor = positive_tensor.view(1, -1)
        negative_tensor = negative_tensor.view(1, -1)
        anchor = model(anchor_tensor.to(device))
        positive = model(positive_tensor.to(device))
        negative = model(negative_tensor.to(device))
        loss = loss_fn(anchor,positive, negative)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * 1 + len(anchor)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for anchor,positive,negative in dataloader:
            anchor_tensor = torch.stack(anchor).float()
            positive_tensor = torch.stack(positive).float()
            negative_tensor = torch.stack(negative).float()
            anchor_tensor = anchor_tensor.view(1, -1)
            positive_tensor = positive_tensor.view(1, -1)
            negative_tensor = negative_tensor.view(1, -1)
            anchor = model(anchor_tensor.to(device))
            positive = model(positive_tensor.to(device))
            negative = model(negative_tensor.to(device))
            test_loss += loss_fn(anchor, positive, negative).item()
            if test_loss < 1:
                correct += 1

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    input_training_filepath = "/mnt/d/npy_anotados_training/"
    input_testing_filepath = "/mnt/d/npy_anotados_testing/"
    all_train_list = []
    all_test_list = []

    for filename in os.listdir(input_training_filepath):
        embeddings = get_dataset_from_npy(filename.replace("_triplets_anotado.npy", ""), input_training_filepath)
        all_train_list.extend(embeddings)

    for filename in os.listdir(input_testing_filepath):
        embeddings = get_dataset_from_npy(filename.replace("_triplets_anotado.npy", ""), input_testing_filepath)
        emb_length = len(embeddings[0]['duplas'][0])
        all_test_list.extend(embeddings)

    combined_training_dataset = ConcatDataset(all_train_list)
    combined_testing_dataset = ConcatDataset(all_test_list)
    train_dataloader = DataLoader(combined_training_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(combined_testing_dataset, batch_size=1, shuffle=True)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork(emb_length).to(device)
    """
    anchor, positive, negative = next(iter(train_dataloader))
    anchor_tensor = torch.stack(anchor).float()
    if anchor_tensor.shape[0] == 7655:
        anchor_tensor = anchor_tensor.view(1, -1)
    positive_tensor = torch.stack(positive).float()
    negative_tensor = torch.stack(negative).float()
    if positive_tensor.shape[0] == 7655:
        positive_tensor = positive_tensor.view(1, -1)
        negative_tensor = negative_tensor.view(1, -1)

    print(f"Nueva entrada: {anchor_tensor.shape}")
    anchor = model(anchor_tensor.to(device))
    positive = model(positive_tensor.to(device))
    negative = model(negative_tensor.to(device))
    print(f"Nueva salida: {anchor.shape}")
    """
    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")


