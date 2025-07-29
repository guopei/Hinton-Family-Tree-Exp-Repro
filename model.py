import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
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
        self.layer5 = nn.Linear(24, 24)
        
    def forward(self, x_1, x_2):
        out_1 = self.layer1_1(x_1)
        out_2 = self.layer1_2(x_2)
        out = torch.cat((out_1, out_2), dim=1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out