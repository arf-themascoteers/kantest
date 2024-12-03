from kan import *
from kan.utils import ex_round
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

df = pd.read_csv("indian_pines.csv").to_numpy()
df_train, df_test = train_test_split(df, train_size=0.9)
train_input = torch.tensor(df_train[:, :-1], dtype=torch.float64).to(device)
train_label = torch.tensor(df_train[:, -1], dtype=torch.long).to(device)
test_input = torch.tensor(df_test[:, :-1], dtype=torch.float64).to(device)
test_label = torch.tensor(df_test[:, -1], dtype=torch.long).to(device)

dataset = {
    "train_input": train_input,
    "train_label": train_label,
    "test_input": test_input,
    "test_label": test_label
}

print(f"Training data shape: {dataset['train_input'].shape}, Labels shape: {dataset['train_label'].shape}")

model = KAN(width=[200, 64, 16], grid=3, k=3, seed=42, device=device)
model(dataset['train_input'])
model.plot()
plt.show()
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001, loss_fn=torch.nn.CrossEntropyLoss())
model.plot()
plt.show()

with torch.no_grad():
    inputs = dataset['test_input']
    outputs = model(inputs)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == dataset['test_label']).sum().item() / len(predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
