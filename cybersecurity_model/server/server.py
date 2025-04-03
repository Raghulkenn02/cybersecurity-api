import flwr as fl
import torch
import torch.nn as nn
import numpy as np

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

# Initialize global model
model = SimpleNN()

def get_parameters():
    return [val.cpu().numpy() for val in model.state_dict().values()]

def set_parameters(parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

# Define Flower strategy
class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_params = super().aggregate_fit(rnd, results, failures)
        if aggregated_params:
            set_parameters(aggregated_params)
            print(f"Round {rnd} updated global model")
        return aggregated_params

# Start the server
fl.server.start_server(
    server_address="localhost:9091",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=CustomStrategy(),
)
