import os
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from Model_training.get_dataset_from_npy import get_dataset_from_npy
from Model_training.NeuralNetwork import NeuralNetwork
from Model_training.NeuralNetwork_Complex import NeuralNetwork_Complex
from torch import nn
from Visualization.loss_function import plot_loss
from Visualization.triplet_visualization import visualize_embeddings_bs1
import psycopg2
import yaml
import pg8000.native 
import socks

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    test_loss = 0
    num_batches = len(dataloader)
    for batch, (anchor, positive, negative) in enumerate(dataloader):
        """
        anchor_tensor = torch.stack(anchor).float()
        positive_tensor = torch.stack(positive).float()
        negative_tensor = torch.stack(negative).float()
        
        anchor_tensor = anchor.view(1, -1)
        positive_tensor = positive.view(1, -1)
        negative_tensor = negative.view(1, -1)
        """
        
        anchor = model(anchor.to(device))
        positive = model(positive.to(device))
        negative = model(negative.to(device))
        loss = loss_fn(anchor,positive, negative)
        test_loss += loss.item()
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()


        if batch % 100 == 0:
            loss, current = loss.item(), batch * 1 + len(anchor)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    test_loss /= num_batches
    return test_loss



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
            """
            anchor_tensor = anchor.view(1, -1)
            positive_tensor = positive.view(1, -1)
            negative_tensor = negative.view(1, -1)
            """
            anchor = model(anchor.to(device))
            positive = model(positive.to(device))
            negative = model(negative.to(device))
            perdida = loss_fn(anchor, positive, negative).item()
            test_loss += perdida

            if perdida < 1:
                correct += 1

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss



if __name__ == '__main__':
    print("a")
    with open('config_ML.yaml', 'r') as file:
        yaml_data = yaml.safe_load(file)
    input_training_filepath = yaml_data['input_training_filepath']
    input_testing_filepath = yaml_data['input_testing_filepath']
    output_filepath = yaml_data['output_filepath']
    mode = yaml_data['mode']
    all_train_list = []
    all_test_list = []
    if mode == "remote":
      s = socks.socksocket()
      s.set_proxy(socks.SOCKS5, "127.0.0.1", 1055)
      s.settimeout(5)
      s.connect(("100.114.94.10", 5432))
      conn = pg8000.native.Connection(
      user=yaml_data['user'],
      password=yaml_data['password'],
      database=yaml_data['dbname'],
      sock=s  
      )
    elif mode =="local":
      conn = pg8000.native.Connection(
      user=yaml_data['user'],
      password=yaml_data['password'],
      database=yaml_data['dbname'],
      host=yaml_data['host'], 
      port=yaml_data['port']          
      )


    for filename in os.listdir(input_training_filepath):
      if filename.endswith(".npy"):
        print("a")
        embeddings = get_dataset_from_npy(filename.replace("_triplets_anotado.npy", ""), input_training_filepath, conn)
        print(filename)
        all_train_list.extend(embeddings)

    for filename in os.listdir(input_testing_filepath):
      if filename.endswith(".npy"):
        embeddings = get_dataset_from_npy(filename.replace("_triplets_anotado.npy", ""), input_testing_filepath, conn)
        print(filename)
        emb_length = len(embeddings[0]['duplas'][0][0])
        all_test_list.extend(embeddings)
    print("Carga terminada")
    combined_training_dataset = ConcatDataset(all_train_list)
    combined_testing_dataset = ConcatDataset(all_test_list)
    train_dataloader = DataLoader(combined_training_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(combined_testing_dataset, batch_size=1, shuffle=True)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork_Complex(emb_length).to(device)
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

    epochs = yaml_data['epochs']
    train_loss_plot = []
    test_loss_plot = []
    for t in range(epochs):
        visualize_embeddings_bs1(model, test_dataloader, device, 30)
        print(f"Epoch {t + 1}\n-------------------------------")
        avg_train_lss = train_loop(train_dataloader, model, loss_fn, optimizer)
        train_loss_plot.append(avg_train_lss)
        avg_test_loss = test_loop(test_dataloader, model, loss_fn)
        test_loss_plot.append(avg_test_loss)

    visualize_embeddings_bs1(model, test_dataloader, device, 30)
    plot_loss(train_loss_plot, test_loss_plot)
    torch.save(model.state_dict(), os.path.join(output_filepath,"pesos_modelo.pt"))
    print("Done!")


