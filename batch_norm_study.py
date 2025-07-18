import torch
import torch.nn as nn
import torch.optim as optim
from data import relationships
from collections import defaultdict
import random
from model import Model
from data import prepare_data
import os

activations = defaultdict(list)

def save_activations_hook(name, module, input, output):
    activations[name].append((input[0].detach(), output.detach()))

def run_once(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    train_name_inputs, train_relation_inputs, train_name_outputs, test_name_inputs, test_relation_inputs, test_name_outputs = prepare_data()

    model = Model()

    # This is how to register a forward hook to a batch norm layer.
    model.layer4.batch_norm.register_forward_hook(lambda module, input, output: save_activations_hook('layer4_output', module, input, output))

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    if os.path.exists(f"model_{random_seed}.pth"):
        print(f"Loading model from {random_seed}")
        model.load_state_dict(torch.load(f"model_{random_seed}.pth"))
    else:    
        print(f"Training model from {random_seed}")
        model.train()
        for i in range(2000):
            optimizer.zero_grad()
            outputs = model(train_name_inputs, train_relation_inputs)
            loss = criterion(outputs, train_name_outputs)
            loss.backward()
            optimizer.step()

        print(f"Saving model to {random_seed}")
        torch.save(model.state_dict(), f"model_{random_seed}.pth")

    model.eval()
    with torch.no_grad():

        test_outputs = model(test_name_inputs, test_relation_inputs)

        input = activations['layer4_output'][0][0]
        output = activations['layer4_output'][0][1]
        print(input.shape, output.shape)
        torch.set_printoptions(precision=2, sci_mode=False)
        
        # See how the input is whitened (max value supressed because of normalization.)
        print(input[0])
        print(output[0])

        test_acc = (sum(test_outputs.argmax(dim=1) == test_name_outputs.argmax(dim=1)) / len(test_outputs))
        print(random_seed, f"{test_acc.item():.2f}")
        return test_acc.item()


if __name__ == "__main__":
    run_once(0)