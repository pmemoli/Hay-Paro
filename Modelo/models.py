import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from transformers import AutoModelForMaskedLM

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

class LogisticRegression(nn.Module):
    def __init__(self, seq_length, embedding_size=768):
        super().__init__()

        self.bert = AutoModelForMaskedLM.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-uncased",
             output_hidden_states=True)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * embedding_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.hidden_states[-1]

        output = self.layer_stack(embeddings)

        return output

class SimpleClassifier(nn.Module):
    def __init__(self, seq_length, embedding_size=768, hidden_size=256):
        super().__init__()

        self.bert = AutoModelForMaskedLM.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-uncased",
             output_hidden_states=True)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * embedding_size, hidden_size),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.hidden_states[-1]

        output = self.layer_stack(embeddings)

        return output

class RnnClassifier(nn.Module):
    def __init__(self, embedding_size=768, hidden_size=128):
        super().__init__()

        self.hidden_size = hidden_size
        self.bert = AutoModelForMaskedLM.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-uncased",
             output_hidden_states=True)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.hidden_states[-1]

        h0 = torch.zeros(2, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(2, batch_size, self.hidden_size).to(device)

        lstm_output, (hn, cn) = self.lstm(embeddings, (h0, c0))
        hn = torch.cat((hn[0, :, :], hn[1, :, :]), dim=1)

        yn = self.linear(hn).squeeze()
        output = self.activation(yn)

        return output

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch_number, (X, y, a) in enumerate(dataloader):
        X, y, a = X.to(device), y.to(device), a.to(device)

        pred = model(X, a)
        if pred.dim() > 1:
            pred = pred.squeeze(dim=1) # [batch size]

        loss = loss_fn(pred, y.float())  

        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()

        if batch_number % 5 == 0:  
            loss, current = loss.item(), (batch_number + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") 

def test_loop(dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    TP, TN, FP, FN = 0, 0, 0, 0

    with torch.no_grad():
        for X, y, a in dataloader:
            X, y, a = X.to(device), y.to(device), a.to(device)

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

    try:
        recall = round(TP / (TP + FN), 2)
        precision = round(TP / (TP + FP), 2)
        accuracy = round(correct / size, 2)
        f1 = round(2 * recall * precision / (recall + precision), 2)

        print(f"\nPrecision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(f"Accuracy: {accuracy}\n")

        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'F1': f1,
        }
    except:
        print('Division by 0!', TP, TN, FP, FN)

