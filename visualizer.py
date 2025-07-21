from data import name_to_index, relation_to_index
from model import Model
import torch
import matplotlib.pyplot as plt
import numpy as np
import math

name_visualization_order = [
    "Christopher",
    "Andrew",
    "Arthur",
    "James",
    "Charles",
    "Colin",
    "Penelope",
    "Christine",
    "Victoria",
    "Jennifer",
    "Margaret",
    "Charlotte",
    "Roberto",
    "Pierro",
    "Emilio",
    "Marco",
    "Tomaso",
    "Alfonso",
    "Maria",
    "Francesca",
    "Lucia",
    "Angela",
    "Gina",
    "Sophia",
]
relation_visualization_order = [
    "has_husband",
    "has_wife",
    "has_son",
    "has_daughter",
    "has_father",
    "has_mother",
    "has_brother",
    "has_sister",
    "has_nephew",
    "has_niece",
    "has_uncle",
    "has_aunt",
]

def draw_weights(weights, random_seed, weights_type):
    # Assume weights is a 1D array of size 12.
    (node_num, input_num) = weights.shape

    half_input_num = input_num // 2
    weight_map = np.zeros((node_num*2*10, half_input_num*10))

    for i in range(node_num):    
        for j in range(input_num):
            # Squre width
            width = int(abs(weights[i, j].item()) * 10)
            height = int(abs(weights[i, j].item()) * 10)
            # Initial position
            top = (i*2 + (j // half_input_num)) * 10
            left = (j % half_input_num) * 10
            # Center the square
            top += (10 - height) // 2
            left += (10 - width) // 2
            # Calculate bottom and right
            bottom = top + height
            right = left + width

            weight_map[top:bottom, left:right] = -1 if weights[i, j].item() < 0 else 1

    plt.imshow(weight_map, cmap='gray', vmin=-1, vmax=1)
    plt.colorbar(label='Weight')
    x_tick_text = name_visualization_order[:half_input_num] if weights_type == "name_weights" else relation_visualization_order[:half_input_num]
    plt.xticks(np.arange(5, half_input_num*10, 10), x_tick_text, rotation=90)
    plt.yticks(np.arange(10, node_num*2*10, 20), np.arange(node_num)+1)
    plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.savefig(f'plots/{weights_type}_{random_seed}.png', dpi=300, bbox_inches='tight')
    plt.close()     



def visualize(random_seed):
    model_weights_path = f"model_weights/model_{random_seed}.pth"
    model_weights = torch.load(model_weights_path)
    model = Model()
    model.load_state_dict(model_weights)
    model.eval()

    name_indexes = [name_to_index[name] for name in name_visualization_order]
    relation_indexes = [relation_to_index[relation] for relation in relation_visualization_order]

    name_weights = model.layer1_1.linear.weight.data
    relation_weights = model.layer1_2.linear.weight.data

    name_weights = name_weights[:, name_indexes]
    relation_weights = relation_weights[:, relation_indexes]
    
    draw_weights(name_weights, random_seed, "name_weights")
    draw_weights(relation_weights, random_seed, "relation_weights")