import random
import pandas as pd
from transformers import top_k_top_p_filtering
import torch.nn.functional as F
from IPython.display import display, HTML
import torch
import copy

BLOCK_SIZE = 128


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

def group_texts(examples):
    # Concatenate all texts.
    ## DEBUG
    # rm_features = [f for f in examples.keys() if f not in ['attention_mask', 'input_ids']]
    # _ = [examples.pop(f) for f in rm_features]
    ##
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy() # the model knows to shift the labels by itself
    return result

def frankenstein(implantee, donor, layer_nums):
    new_state_dict = implantee.state_dict().copy()
    donor_state_dict = donor.state_dict().copy()
    monster = copy.copy(implantee)
    for layer_num in layer_nums:
        keys = [key for key in implantee.state_dict().keys()
                if key.startswith(f'transformer.h.{layer_num}')
                or key.startswith(f'roberta.encoder.layer.{layer_num}')]
        for key in keys:
            new_state_dict[key] = donor_state_dict[key]
    monster.load_state_dict(new_state_dict)
    return monster

def generate(model, prompt, tokenizer, top_k=60, temp=1):
    device = model.device
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=top_k, temperature=temp)
    generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]
    return generated

def predict_next(model, sequence, tokenizer):
    device = model.device
    input_ids = tokenizer.encode(sequence, return_tensors="pt").to(device)
    next_token_logits = model(input_ids.to(device))[0][:, -1, :]
    filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    top_tokens = probs.sort()[1][-1].flip(-1)[0:5]
    top_probs = probs.sort()[0][-1].flip(-1)[0:5]
    predictions = ['{2:.3}:\t{0}{1}'.format(sequence, tokenizer.decode(top_tokens[i]), top_probs[i]) for i in range(5)]
    return predictions

def mlm(model, sequence, tokenizer):
    # {tokenizer.mask_token}
    device = model.device
    input = tokenizer.encode(sequence, return_tensors="pt").to(device)
    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    token_logits = model(input)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    filtered_mask_token_logits = top_k_top_p_filtering(mask_token_logits, top_k=50, top_p=1.0)
    probs = F.softmax(filtered_mask_token_logits, dim=-1)
    top_tokens = probs.sort()[1][-1].flip(-1)[0:5]
    top_probs = probs.sort()[0][-1].flip(-1)[0:5]
    results = ['{0:.3}:\t{1}'.format(top_probs[i], sequence.replace(tokenizer.mask_token,
               '==' + tokenizer.decode(top_tokens[i]) + ' ==')) for i in range(5)]
    return results
