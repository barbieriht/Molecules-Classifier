import os
import pandas as pd
import numpy as np
import math
import json
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
from itertools import product
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
    
class smiles_coder:
    def __init__(self):
        self.char_set = set([' '])
        self.char_to_int = None
        self.int_to_char = None
        self.fitted = False

        if os.path.exists('smiles_vocab.npz'):
            self.load('smiles_vocab.npz')

    def fit(self, smiles_data, max_length = 150):
        for i in tqdm(range(len(smiles_data))):
            smiles_data[i] = smiles_data[i].ljust(max_length)
            self.char_set = self.char_set.union(set(smiles_data[i]))
        self.max_length = max_length
        self.n_class = len(self.char_set)
        self.char_to_int = dict((c, i) for i, c in enumerate(self.char_set))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.char_set))
        self.fitted = True

    def transform(self, smiles_data):
        if not self.fitted:
            raise ValueError('smiles coder is not fitted')
        m = []
        for i in tqdm(range(len(smiles_data))):
            smiles_data[i] = smiles_data[i].ljust(self.max_length)
            chars = smiles_data[i]
            l = np.zeros((self.max_length,))
            for t, char in enumerate(chars):
                if t >= self.max_length:
                    break
                else:
                    if char in self.char_set:
                        l[t] = self.char_to_int[char]
            m.append(l)
        return np.array(m)

    def char_to_int(self):
        return self.char_to_int

    def save(self, save_path):
        np.savez(save_path, char_set = self.char_set, char_to_int=self.char_to_int,
                            max_length = self.max_length, n_class = len(self.char_set))

    def load(self, save_path):
        saved = np.load(save_path, allow_pickle=True)
        self.char_set = saved['char_set'].tolist()
        self.char_to_int = saved['char_to_int'].tolist()
        self.max_length = saved['max_length'].tolist()
        self.n_class = len(self.char_set)
        self.fitted = True

replace_dict = {'Ag':'D', 'Al':'E', 'Ar':'G', 'As':'J', 'Au':'Q', 'Ba':'X', 'Be':'Y',
                'Br':'f', 'Ca':'h', 'Cd':'j', 'Ce':'k', 'Cl':'m', 'Cn':'p', 'Co':'q',
                'Cr':'v', 'Cu':'w', 'Fe':'x', 'Hg':'y', 'Ir':'z', 'La':'!', 'Mg':'$',
                'Mn':'¬', 'Mo':'&', 'Na':'_', 'Ni':'£', 'Pb':'¢', 'Pt':'?', 'Ra':'ª',
                'Ru':'º', 'Sb':';', 'Sc':':', 'Se':'>', 'Si':'<', 'Sn':'§', 'Sr':'~',
                'Te':'^', 'Tl':'|', 'Zn':'{', '@@':'}'}

def preprocessing_data(molecules, replacement, saving=False):

    if not saving:
        molecules = pd.Series(molecules)
    else:
        molecules = pd.Series([mol for mol in molecules])

    for pattern, repl in replacement.items():
        molecules = molecules.str.replace(pattern, repl, regex=False)

    max_length = len(max(molecules, key=len))

    return molecules, max_length

def get_hot_smiles(file_name):
    if type(file_name) == list:
        this_df = pd.concat([pd.read_csv(fn, header=0) for fn in file_name])
    else:
        this_df = pd.read_csv(file_name, header=0)
    molecules = this_df['smile'].to_list()
    classes = this_df['classe'].to_list()

    classes_hot_array = np.array([])

    if not os.path.exists('classes_vocab.json'):
        unique_elements = list(set(classes))


        for classe in classes:
            classes_hot_array = np.append(classes_hot_array, unique_elements.index(classe))

        classes_vocab = {}
        for i, elem in enumerate(unique_elements):
            classes_vocab[elem] = i

        with open('classes_vocab.json', 'w') as f:
            json.dump(classes_vocab, f)
            f.close()
    else:
        with open('classes_vocab.json', 'r') as f:
            classes_vocab = json.load(f)
            f.close()

        for classe in classes:
            classes_hot_array = np.append(classes_hot_array, classes_vocab[classe])

    processed_molecules, smiles_max_length = preprocessing_data(molecules, replace_dict)

    ######## TURNING TO ONE HOT ########
    coder = smiles_coder()
    if not os.path.exists('smiles_vocab.npz'):
        coder.fit(processed_molecules, smiles_max_length)
        coder.save('smiles_vocab.npz')
    smiles_hot_arrays = coder.transform(processed_molecules)

    return smiles_hot_arrays, classes_hot_array, coder, classes_vocab

class DrugLikeMolecules(Dataset):
    def __init__(self, file_path=None):
        self.smiles, self.classes, self.coder, self.classes_vocab = get_hot_smiles(file_path)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        sml = self.smiles[idx]
        labels = self.classes[idx]

        return sml, labels

class SmileClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SmileClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.rnn1 = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.rnn2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x.to(device, dtype=torch.long))
        output1, _ = self.rnn1(embedded)
        output2, _ = self.rnn2(output1)
        
        # Assuming you want to use the final hidden state of the last LSTM layer
        last_hidden = output2[:, -1, :]
        
        logits = self.fc(last_hidden)
        return logits

if __name__ == "__main__":
    full_dataset = DrugLikeMolecules(file_path=['train.csv', 'test.csv', 'val.csv'])

    cross_val_dataset = DrugLikeMolecules(file_path=['train.csv', 'val.csv'])
    test_dataset = DrugLikeMolecules(file_path='test.csv')

    vocab_size = len(full_dataset.coder.char_set)
    num_classes = len(set(full_dataset.classes))

    embedding_dim = full_dataset.coder.max_length

    num_folds = 5
    early_stop_patience = 5

    hidden_dim = 128
    batch_size = 128
    num_epochs = 250
    learning_rate = 0.01

    # Create the model
    model = SmileClassifier(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses_per_fold = []
    val_losses_per_fold = []

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(cross_val_dataset)):
        print(f"\nFold {fold + 1}/{num_folds}")

        train_fold_dataset = Subset(cross_val_dataset, train_idx)
        valid_fold_dataset = Subset(cross_val_dataset, valid_idx)

        train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_fold_dataset, batch_size=batch_size, shuffle=False)

        train_losses_per_epoch = []
        val_losses_per_epoch = []

        best_val_loss = float('inf')
        consecutive_val_no_improvement = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            total_samples = 0

            for inputs, classes in train_loader:
                inputs, targets = inputs.to(device, dtype=torch.long), classes.to(device, dtype=torch.long)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(inputs)
                total_samples += len(inputs)


            avg_loss = total_loss / total_samples
            train_losses_per_epoch.append(avg_loss)

            # Evaluate on the validation set after every epoch
            model.eval()
            total_val_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for inputs, classes in valid_loader:
                    inputs, targets = inputs.to(device, dtype=torch.long), classes.to(device, dtype=torch.long)

                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)

                    total_val_loss += val_loss.item() * len(inputs)
                    total_val_samples += len(inputs)

            avg_val_loss = total_val_loss / total_val_samples
            val_losses_per_epoch.append(avg_val_loss)

            print(f"Fold {fold + 1}/{num_folds}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}", end='\r')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                consecutive_val_no_improvement = 0
            else:
                consecutive_val_no_improvement += 1

            if consecutive_val_no_improvement >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1} due to no improvement in val losses.")
                break 

        # Save the plot for this fold
        plt.figure(figsize=(10, 6))
        epochs = np.arange(1, len(val_losses_per_epoch)+1)
        plt.plot(epochs, train_losses_per_epoch, label='Train Loss')
        plt.plot(epochs, val_losses_per_epoch, label='Val Loss')
        plt.title(f"Fold {fold + 1}/{num_folds} Losses per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"fold_{fold + 1}_loss_plot.png")
        plt.close()

        # Save losses for later analysis
        train_losses_per_fold.append(train_losses_per_epoch)
        val_losses_per_fold.append(val_losses_per_epoch)

    plt.figure(figsize=(12, 8))
    for fold, (train_losses, val_losses) in enumerate(zip(train_losses_per_fold, val_losses_per_fold), start=1):
        plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label=f'Fold {fold} Train')
        plt.plot(np.arange(1, len(train_losses) + 1), val_losses, label=f'Fold {fold} Val')

    plt.title("Training and Validation Losses Across Folds")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("overall_loss_plot.png")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Test the model on the holdout test set
    model.eval()
    total_test_loss = 0.0
    total_test_samples = 0

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, classes in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.long), classes.to(device, dtype=torch.long)

            outputs = model(inputs)
            test_loss = criterion(outputs, targets)

            total_test_loss += test_loss.item() * len(inputs)
            total_test_samples += len(inputs)


            # Convert logits to predicted labels
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(targets.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Calculate and print the average test loss
    avg_test_loss = total_test_loss / total_test_samples
    print(f"Holdout Test Loss: {avg_test_loss:.4f}")

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision_micro = precision_score(true_labels, predicted_labels, average='micro')
    precision_macro = precision_score(true_labels, predicted_labels, average='macro')
    precision_weighted = precision_score(true_labels, predicted_labels, average='weighted')
    recall_micro = recall_score(true_labels, predicted_labels, average='micro')
    recall_macro = recall_score(true_labels, predicted_labels, average='macro')
    recall_weighted = recall_score(true_labels, predicted_labels, average='weighted')
    f1_micro = f1_score(true_labels, predicted_labels, average='micro')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')

    # Plot and save the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Plot and save precision, recall, and F1 score
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Micro', 'Macro', 'Weighted']
    precision_scores = [precision_micro, precision_macro, precision_weighted]
    recall_scores = [recall_micro, recall_macro, recall_weighted]
    f1_scores = [f1_micro, f1_macro, f1_weighted]

    ax.bar(metrics, precision_scores, width=0.2, label='Precision', align='center', alpha=0.7)
    ax.bar(metrics, recall_scores, width=0.2, label='Recall', align='edge', alpha=0.7)
    ax.bar(metrics, f1_scores, width=0.2, label='F1 Score', align='edge', alpha=0.7)

    ax.set_ylabel('Scores')
    ax.set_title('Precision, Recall, and F1 Score')
    ax.legend(loc='upper right')

    plt.savefig('precision_recall_f1.png')
    plt.show()