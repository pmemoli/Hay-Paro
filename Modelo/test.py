
#%% nn imports
import torch
from torch import nn
import numpy as np
import models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from collections import defaultdict

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

_ = torch.manual_seed(42)
print(torch.__version__)
print(torch.version.cuda)
print(f"Using {device} device")

#%% Dataset setup
df_test = pd.read_csv('../Data/test.csv')
titulos_test, labels_test = list(df_test['texto']), list(df_test['paro'])
relacionado_test = list(df_test['relacionado'])

class TitulosDataset(Dataset):
    def __init__(self, titulos, labels, relacionados):
        super().__init__()
        self.titulos = titulos
        self.labels = labels
        self.relacionados = relacionados

    def __len__(self):
        return len(self.titulos)
    
    def __getitem__(self, idx):
        return self.titulos[idx], self.labels[idx], self.relacionados[idx]

#%% Train Model Function based on dataset and model name
test_dataset = TitulosDataset(titulos_test, labels_test, relacionado_test)

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

inputs_test = tokenizer(test_dataset.titulos, return_tensors="pt", padding=True)
max_length = inputs_test.input_ids.size(1)

# Set up data loader
batch_size = 128

def collate_fn(batch):
    titulos, labels, relacionados = zip(*batch)
    labels = torch.IntTensor(labels)
    relacionados = torch.IntTensor(relacionados)

    inputs = tokenizer(titulos, return_tensors="pt", padding="max_length",
                    max_length=max_length, truncation=True)

    return inputs['input_ids'], labels, inputs['attention_mask'], relacionados

test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, collate_fn=collate_fn)

#%% Load model and test parameters
def test_loop(dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    TP, TN, FP, FN = 0, 0, 0, 0

    relacionados = 0
    relacionados_correct = 0

    no_relacionados = 0
    no_relacionados_correct = 0

    with torch.no_grad():
        for X, y, a, r in dataloader:
            X, y, a, r = X.to(device), y.to(device), a.to(device), r.to(device)

            pred = model(X, a)
            if pred.dim() > 1:
                pred = pred.squeeze(dim=1) # [batch size]

            test_loss += loss_fn(pred, y.float()).item()

            pred_labels = (pred >= 0.5).float() 
            correct += (pred_labels == y.float()).sum().item()

            TP += ((pred_labels == 1) & (y.float() == 1)).sum().item()
            TN += ((pred_labels == 0) & (y.float() == 0)).sum().item()
            FP += ((pred_labels == 1) & (y.float() == 0)).sum().item()
            FN += ((pred_labels == 0) & (y.float() == 1)).sum().item()

            relacionados += (r.float() == 1).sum().item()
            relacionados_correct += ((pred_labels == 1) & (r.float() == 1)).sum().item()

            no_relacionados += (r.float() == 0).sum().item()
            no_relacionados_correct += ((pred_labels == 0) & (r.float() == 0)).sum().item()

        recall = round(TP / (TP + FN), 2)
        precision = round(TP / (TP + FP), 2)
        accuracy = round(correct / size, 2)
        f1 = round(2 * recall * precision / (recall + precision), 2)

        relacionado_accuracy = relacionados_correct / relacionados
        no_relacionado_accuracy = no_relacionados_correct / no_relacionados

        return {
            'precision': precision,
            'recall': recall,
            'F1': f1,
            'accuracy': accuracy,
            'rel_accuracy': relacionado_accuracy,
            'no_rel_accuracy': no_relacionado_accuracy,
        }

#%%
model = models.RnnClassifier().to(device)
state_dict = torch.load('saved_models/final_model')
model.load_state_dict(state_dict)

loss_fn = nn.BCELoss()

#%%
result = test_loop(test_dataloader, model, loss_fn)

print(result)

# %%
