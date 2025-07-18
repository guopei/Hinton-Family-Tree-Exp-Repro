import torch
import torch.nn as nn
import torch.optim as optim
from data import prepare_data
import random
from model import Model
    

def run_once(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    train_name_inputs, train_relation_inputs, train_name_outputs, test_name_inputs, test_relation_inputs, test_name_outputs = prepare_data()

    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    for i in range(2000):
        optimizer.zero_grad()
        outputs = model(train_name_inputs, train_relation_inputs)
        loss = criterion(outputs, train_name_outputs)
        loss.backward()
        optimizer.step()
        # if i % 100 == 0:
        #     print(i, loss.item())

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_name_inputs, test_relation_inputs)
        test_acc = (sum(test_outputs.argmax(dim=1) == test_name_outputs.argmax(dim=1)) / len(test_outputs))
        train_acc = (sum(outputs.argmax(dim=1) == train_name_outputs.argmax(dim=1)) / len(train_name_outputs))
        print(random_seed, f"{test_acc.item():.2f}", f"{train_acc.item():.2f}")
        return test_acc.item()


if __name__ == "__main__":
    total_run = 50
    test_accs = []
    for i in range(total_run):
        test_acc = run_once(i)
        test_accs.append(test_acc)
    print(sum(test_accs) / total_run)