import torch
import torch.nn as nn
import torch.optim as optim
from data import relationships
from collections import defaultdict
import random

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(output_size)
        
    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        out = self.batch_norm(out)
        return out
        
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1_1 = Layer(24, 6)
        self.layer1_2 = Layer(12, 6)

        self.layer2 = Layer(12, 6)
        self.layer3 = Layer(6, 12)
        self.layer4 = Layer(12, 24)
        
    def forward(self, x_1, x_2):
        out_1 = self.layer1_1(x_1)
        out_2 = self.layer1_2(x_2)
        out = torch.cat((out_1, out_2), dim=1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def prepare_data():
    train_num = 100
    random.shuffle(relationships)

    names = set([name for name, _, _ in relationships])
    names = list(names)
    names.sort()
    
    relations = set([relation for _, relation, _ in relationships])
    relations = list(relations)
    relations.sort()

    name_to_index = {name: i for i, name in enumerate(names)}
    relation_to_index = {relation: i for i, relation in enumerate(relations)}

    name_inputs = torch.zeros(len(relationships), len(names))
    relation_inputs = torch.zeros(len(relationships), len(relations))
    name_outputs = torch.zeros(len(relationships), len(names))

    for i, (name_input, relation_input, output) in enumerate(relationships):
        name_inputs[i][name_to_index[name_input]] = 1
        relation_inputs[i][relation_to_index[relation_input]] = 1
        for output_name in output:
            name_outputs[i][name_to_index[output_name]] = 1

    train_name_inputs = name_inputs[:train_num]
    train_relation_inputs = relation_inputs[:train_num]
    train_name_outputs = name_outputs[:train_num]

    test_name_inputs = name_inputs[train_num:]
    test_relation_inputs = relation_inputs[train_num:]
    test_name_outputs = name_outputs[train_num:]

    return train_name_inputs, train_relation_inputs, train_name_outputs, test_name_inputs, test_relation_inputs, test_name_outputs
    

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