import torch
import random

relationships = [
    ("Penelope", "has_husband", ["Christopher"]),
    ("Christine", "has_husband", ["Andrew"]),
    ("Margaret", "has_husband", ["Arthur"]),
    ("Victoria", "has_husband", ["James"]),
    ("Jennifer", "has_husband", ["Charles"]),
    ("Maria", "has_husband", ["Roberto"]),
    ("Francesca", "has_husband", ["Pierro"]),
    ("Gina", "has_husband", ["Emilio"]),
    ("Lucia", "has_husband", ["Marco"]),
    ("Angela", "has_husband", ["Tomaso"]),
    ("Christopher", "has_wife", ["Penelope"]),
    ("Andrew", "has_wife", ["Christine"]),
    ("Arthur", "has_wife", ["Margaret"]),
    ("James", "has_wife", ["Victoria"]),
    ("Charles", "has_wife", ["Jennifer"]),
    ("Roberto", "has_wife", ["Maria"]),
    ("Pierro", "has_wife", ["Francesca"]),
    ("Emilio", "has_wife", ["Gina"]),
    ("Marco", "has_wife", ["Lucia"]),
    ("Tomaso", "has_wife", ["Angela"]),
    ("Christopher", "has_son", ["Arthur"]),
    ("Penelope", "has_son", ["Arthur"]),
    ("Andrew", "has_son", ["James"]),
    ("Christine", "has_son", ["James"]),
    ("Victoria", "has_son", ["Colin"]),
    ("James", "has_son", ["Colin"]),
    ("Roberto", "has_son", ["Emilio"]),
    ("Maria", "has_son", ["Emilio"]),
    ("Lucia", "has_son", ["Alfonso"]),
    ("Marco", "has_son", ["Alfonso"]),
    ("Pierro", "has_son", ["Tomaso"]),
    ("Francesca", "has_son", ["Tomaso"]),
    ("Christopher", "has_daughter", ["Victoria"]),
    ("Penelope", "has_daughter", ["Victoria"]),
    ("Andrew", "has_daughter", ["Jennifer"]),
    ("Christine", "has_daughter", ["Jennifer"]),
    ("Victoria", "has_daughter", ["Charlotte"]),
    ("James", "has_daughter", ["Charlotte"]),
    ("Roberto", "has_daughter", ["Lucia"]),
    ("Maria", "has_daughter", ["Lucia"]),
    ("Lucia", "has_daughter", ["Sophia"]),
    ("Marco", "has_daughter", ["Sophia"]),
    ("Pierro", "has_daughter", ["Angela"]),
    ("Francesca", "has_daughter", ["Angela"]),
    ("Arthur", "has_father", ["Christopher"]),
    ("Victoria", "has_father", ["Christopher"]),
    ("James", "has_father", ["Andrew"]),
    ("Jennifer", "has_father", ["Andrew"]),
    ("Colin", "has_father", ["James"]),
    ("Charlotte", "has_father", ["James"]),
    ("Emilio", "has_father", ["Roberto"]),
    ("Lucia", "has_father", ["Roberto"]),
    ("Alfonso", "has_father", ["Marco"]),
    ("Sophia", "has_father", ["Marco"]),
    ("Marco", "has_father", ["Pierro"]),
    ("Angela", "has_father", ["Pierro"]),
    ("Arthur", "has_mother", ["Penelope"]),
    ("Victoria", "has_mother", ["Penelope"]),
    ("James", "has_mother", ["Christine"]),
    ("Jennifer", "has_mother", ["Christine"]),
    ("Colin", "has_mother", ["Victoria"]),
    ("Charlotte", "has_mother", ["Victoria"]),
    ("Emilio", "has_mother", ["Maria"]),
    ("Lucia", "has_mother", ["Maria"]),
    ("Alfonso", "has_mother", ["Lucia"]),
    ("Sophia", "has_mother", ["Lucia"]),
    ("Marco", "has_mother", ["Francesca"]),
    ("Angela", "has_mother", ["Francesca"]),
    ("Colin", "has_uncle", ["Arthur", "Charles"]),
    ("Charlotte", "has_uncle", ["Arthur", "Charles"]),
    ("Alfonso", "has_uncle", ["Emilio", "Tomaso"]),
    ("Sophia", "has_uncle", ["Emilio", "Tomaso"]),
    ("Colin", "has_aunt", ["Jennifer", "Margaret"]),
    ("Charlotte", "has_aunt", ["Jennifer", "Margaret"]),
    ("Sophia", "has_aunt", ["Angela", "Gina"]),
    ("Alfonso", "has_aunt", ["Angela", "Gina"]),
    ("Jennifer", "has_niece", ["Charlotte"]),
    ("Arthur", "has_niece", ["Charlotte"]),
    ("Margaret", "has_niece", ["Charlotte"]),
    ("Charles", "has_niece", ["Charlotte"]),
    ("Angela", "has_niece", ["Sophia"]),
    ("Emilio", "has_niece", ["Sophia"]),
    ("Gina", "has_niece", ["Sophia"]),
    ("Tomaso", "has_niece", ["Sophia"]),
    ("Arthur", "has_nephew", ["Colin"]),
    ("Jennifer", "has_nephew", ["Colin"]),
    ("Margaret", "has_nephew", ["Colin"]),
    ("Charles", "has_nephew", ["Colin"]),
    ("Angela", "has_nephew", ["Alfonso"]),
    ("Emilio", "has_nephew", ["Alfonso"]),
    ("Gina", "has_nephew", ["Alfonso"]),
    ("Tomaso", "has_nephew", ["Alfonso"]),
    ("Victoria", "has_brother", ["Arthur"]),
    ("Jennifer", "has_brother", ["James"]),
    ("Charlotte", "has_brother", ["Colin"]),
    ("Lucia", "has_brother", ["Emilio"]),
    ("Sophia", "has_brother", ["Alfonso"]),
    ("Angela", "has_brother", ["Marco"]),
    ("Arthur", "has_sister", ["Victoria"]),
    ("James", "has_sister", ["Jennifer"]),
    ("Colin", "has_sister", ["Charlotte"]),
    ("Emilio", "has_sister", ["Lucia"]),
    ("Alfonso", "has_sister", ["Sophia"]),
    ("Marco", "has_sister", ["Angela"]),
]

names = sorted(list(set([name for name, _, _ in relationships])))
relations = sorted(list(set([relation for _, relation, _ in relationships])))

name_to_index = {name: i for i, name in enumerate(names)}
relation_to_index = {relation: i for i, relation in enumerate(relations)}

train_num = 100

def prepare_data():
    random.shuffle(relationships)
    
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

    # Training set soft labels

    # train_name_outputs[train_name_outputs == 1] = 0.8
    # train_name_outputs[train_name_outputs == 0] = 0.2

    return train_name_inputs, train_relation_inputs, train_name_outputs, test_name_inputs, test_relation_inputs, test_name_outputs