import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = SimpleNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        for epoch in range(1):  # Train for 1 epoch
            for images, labels in train_loader:
                images, labels = images.view(-1, 28 * 28), labels
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(config), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.view(-1, 28 * 28), labels
                outputs = self.model(images)
                loss += self.loss_fn(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()

        return loss / len(val_loader), len(val_loader.dataset), {"accuracy": correct / len(val_loader.dataset)}

# Start client
fl.client.start_client(
    server_address="localhost:9091",
    client=FlowerClient().to_client(),
)
