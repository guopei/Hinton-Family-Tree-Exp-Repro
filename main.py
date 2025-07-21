import torch
import torch.nn as nn
import torch.optim as optim
from data import prepare_data
import random
from model import Model
from torcheval.metrics import MultilabelAccuracy


def run_once(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    train_name_inputs, train_relation_inputs, train_name_outputs, test_name_inputs, test_relation_inputs, test_name_outputs = prepare_data()

    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    metric = MultilabelAccuracy()
    
    model.train()
    for i in range(2000):
        optimizer.zero_grad()
        outputs = model(train_name_inputs, train_relation_inputs)
        loss = criterion(outputs, train_name_outputs)
        loss.backward()
        optimizer.step()
        # metric.update(outputs.detach(), train_name_outputs.detach())
        # if i % 100 == 0:
        #     print(i, loss.item(), metric.compute().item())
        # metric.reset()
    
    model.eval()
    metric.reset()
    with torch.no_grad():
        test_outputs = model(test_name_inputs, test_relation_inputs)
        metric.update(test_outputs.detach(), test_name_outputs.detach())
        print(random_seed, metric.compute().item())
        return metric.compute().item()


if __name__ == "__main__":
    total_run = 50
    test_accs = []
    for i in range(total_run):
        test_acc = run_once(i)
        test_accs.append(test_acc)
    print(sum(test_accs) / total_run)