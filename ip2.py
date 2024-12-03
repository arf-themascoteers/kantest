import torch
from kan import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import moviepy.video.io.ImageSequenceClip

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

print(device)


def load_iris_dataset():
  df = pd.read_csv("indian_pines.csv").to_numpy()
  df_train, df_test = train_test_split(df, train_size=0.9)
  train_input = torch.tensor(df_train[:, :-1], dtype=torch.float32).to(device)
  train_label = torch.tensor(df_train[:, -1], dtype=torch.long).to(device)
  test_input = torch.tensor(df_test[:, :-1], dtype=torch.float32).to(device)
  test_label = torch.tensor(df_test[:, -1], dtype=torch.long).to(device)

  dataset = {
    "train_input": train_input,
    "train_label": train_label,
    "test_input": test_input,
    "test_label": test_label
  }

  return dataset


iris_dataset = load_iris_dataset()

model = KAN(width=[200, 5, 16], grid=5, k=3, seed=0, device=device)

model(iris_dataset['train_input'])
#model.plot()


def train_acc():
  return torch.mean((torch.argmax(model(iris_dataset['train_input']), dim=1) == iris_dataset['train_label']).float())


def test_acc():
  return torch.mean((torch.argmax(model(iris_dataset['test_input']), dim=1) == iris_dataset['test_label']).float())

image_folder = 'video_img'
results = model.fit(iris_dataset, opt="Adam", metrics=(train_acc, test_acc),
                    loss_fn=torch.nn.CrossEntropyLoss(), steps=1, lamb=0.01, lamb_entropy=10., save_fig=True,
                    img_folder=image_folder)

print(results['train_acc'][-1], results['test_acc'][-1])
