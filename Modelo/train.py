#%%
# %load_ext autoreload
# %autoreload 2

#%% nn imports
import torch
import pickle
from torch import nn
import numpy as np
import models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from collections import defaultdict

# Function to create a nested defaultdict
def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

_ = torch.manual_seed(42)
print(torch.__version__)
print(torch.version.cuda)
print(f"Using {device} device")

#%% Dataset setup
df_basic_train = pd.read_csv('../Data/basic_train.csv')
df_oversample_train = pd.read_csv('../Data/oversample_train.csv')
df_gpt_oversample_train = pd.read_csv('../Data/balance_gpt_train.csv')

df_validation = pd.read_csv('../Data/validation.csv')

df_basic_train = pd.concat([df_basic_train, df_validation], ignore_index=True)
df_oversample_train = pd.concat([df_oversample_train, df_validation], ignore_index=True)
df_gpt_oversample_train = pd.concat([df_gpt_oversample_train, df_validation], ignore_index=True)

titulos_validation, labels_validation = list(df_validation['texto']), list(df_validation['paro'])

class TitulosDataset(Dataset):
    def __init__(self, titulos, labels):
        super().__init__()
        self.titulos = titulos
        self.labels = labels

    def __len__(self):
        return len(self.titulos)
    
    def __getitem__(self, idx):
        return self.titulos[idx], self.labels[idx]

#%% Train Model Function based on dataset and model name
model_epochs = {
    # 'simple_256': 70,
    'lstm': 1100,
}

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

def train_model(model_name, train_dataset, validation_dataset, dataset_name,
                starting_model_path=None):
    # Generate dataloader
    inputs_train = tokenizer(train_dataset.titulos, return_tensors="pt", padding=True)
    inputs_validation = tokenizer(validation_dataset.titulos, return_tensors="pt", padding=True)

    max_length = max([inputs_train.input_ids.size(1),
                    inputs_validation.input_ids.size(1)])

    if model_name == 'logistic_reg':
        model = models.LogisticRegression(max_length).to(device)
    elif model_name == 'simple_256':
        model = models.SimpleClassifier(max_length, hidden_size=128).to(device)
    elif model_name == 'lstm':
        model = models.RnnClassifier().to(device)

    # Load starting model if there is any
    name_key = ''
    if starting_model_path != None:
        state_dict = torch.load(starting_model_path)
        model.load_state_dict(state_dict)
        name_key = 'cont'

    # Set up data loader
    batch_size = 128

    def collate_fn(batch):
        titulos, labels = zip(*batch)
        labels = torch.IntTensor(labels)

        inputs = tokenizer(titulos, return_tensors="pt", padding="max_length",
                        max_length=max_length, truncation=True)

        return inputs['input_ids'], labels, inputs['attention_mask']

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size, shuffle=True, collate_fn=collate_fn)

    # Set up training 
    loss_fn = nn.BCELoss()

    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Starts proper training 
    epochs = model_epochs[model_name]
    results = []
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        models.train_loop(train_dataloader, model, loss_fn, optimizer)
        # result = models.test_loop(validation_dataloader, model, loss_fn)
        # results.append(result)

        if t > 0 and t % 300 == 0:
            torch.save(model.state_dict(),
                       f"saved_models/{model_name}_{t}_{dataset_name}_{name_key}")

    torch.save(model.state_dict(),
               f"saved_models/{model_name}_{dataset_name}_{name_key}_Final")

    print(f'\nDone Testing! Saved {model_name} {dataset_name}')

    return results

#%% Run all models
train_dfs = {
    'gpt': df_gpt_oversample_train, 
    # 'oversample': df_oversample_train,
    # 'basic': df_basic_train
}

initial_models = {
    'gpt': 'saved_models/lstm_gpt', 
    'oversample': 'saved_models/lstm_oversample', 
    'basic': 'saved_models/lstm_basic', 
}

results = recursive_defaultdict()
for model_name in model_epochs.keys():
    for df_train_key in train_dfs.keys():
        print(f'Training {model_name} for the {df_train_key} dataset...')
        df_train = train_dfs[df_train_key]
        titulos_train, labels_train = list(df_train['texto']), list(df_train['paro'])

        train_dataset = TitulosDataset(titulos_train, labels_train)
        validation_dataset = TitulosDataset(titulos_validation, labels_validation)

        result = train_model(model_name, train_dataset, validation_dataset, df_train_key)

        results[model_name][df_train_key] = result

#%% Save results at each epoch for graphing
# print(results)

# results_dict = defaultdict_to_dict(results)

# train_graphs_path = 'train_graphs/'
# with open(train_graphs_path + 'resultado_lstm_continuacion.pkl', 'wb') as fp:
#     pickle.dump(results_dict, fp)
