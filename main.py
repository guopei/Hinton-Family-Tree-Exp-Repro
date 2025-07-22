import torch
import torch.nn as nn
import torch.optim as optim
from data import prepare_data
import random
from model import Model
from torcheval.metrics import MultilabelAccuracy
from visualizer import visualize
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--visualize", action="store_true")
parser.add_argument("-s", "--save_model", action="store_true")
args = parser.parse_args()

def run_once(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    train_name_inputs, train_relation_inputs, train_name_outputs, test_name_inputs, test_relation_inputs, test_name_outputs = prepare_data()

    model = Model()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    metric = MultilabelAccuracy()
    
    model.train()
    for i in range(1500):
        optimizer.zero_grad()
        outputs = model(train_name_inputs, train_relation_inputs)
        loss = criterion(outputs, train_name_outputs)
        loss.backward()
        optimizer.step()
    
    model.eval()
    metric.reset()
    with torch.no_grad():
        test_outputs = model(test_name_inputs, test_relation_inputs)
        metric.update(test_outputs.detach(), test_name_outputs.detach())
        test_acc = metric.compute().item()
        print(f"random_seed: {random_seed}, test_acc: {test_acc}")
        if args.save_model:
            torch.save(model.state_dict(), f"model_weights/model_{random_seed}.pth")
        return test_acc


if __name__ == "__main__":
    os.makedirs("model_weights", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    total_run = 50
    print(f"Training and evaluating with {total_run} random seeds")
    test_accs = []
    total_perfect_accs = 0
    for i in range(total_run):
        test_acc = run_once(i)
        test_accs.append(test_acc)
        if args.visualize:
            visualize(i)
        if test_acc > 0.99:
            total_perfect_accs += 1
    print(f"Average test accuracy: {sum(test_accs) / total_run}")
    print(f"Total perfect accuracies percentage: {total_perfect_accs / total_run}")