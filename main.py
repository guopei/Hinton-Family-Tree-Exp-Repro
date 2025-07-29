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
from torch.optim.lr_scheduler import LambdaLR

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--visualize", action="store_true")
parser.add_argument("-s", "--save_model", action="store_true")
args = parser.parse_args()

def linear_warmup(step, warmup_steps):
    return min(1.0, step / warmup_steps)

train_epochs = 4000

def run_once(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    train_name_inputs, train_relation_inputs, train_name_outputs, test_name_inputs, test_relation_inputs, test_name_outputs = prepare_data()

    model = Model()
    optimizer = optim.AdamW(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()
    metric = MultilabelAccuracy()
    # Linear warmup
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: linear_warmup(step, train_epochs // 10))

    model.train()
    for i in range(train_epochs):
        optimizer.zero_grad()

        shuffled_indices = torch.randperm(len(train_name_inputs))
        outputs = model(train_name_inputs[shuffled_indices], train_relation_inputs[shuffled_indices])
        loss = criterion(outputs, train_name_outputs[shuffled_indices])
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    
    with torch.no_grad():
        test_outputs = model(test_name_inputs, test_relation_inputs)
        metric.reset()
        metric.update(test_outputs.detach(), test_name_outputs.detach())
        test_acc = metric.compute().item()
        metric.reset()
        train_outputs = model(train_name_inputs, train_relation_inputs)
        metric.update(train_outputs.detach(), train_name_outputs.detach())
        train_acc = metric.compute().item()

        print(f"random_seed: {random_seed:02d}, test_acc: {test_acc:.2f}, train_acc: {train_acc:.2f}, loss: {loss.item():.4f}")
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