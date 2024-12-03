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
  # Load iris dataset
  iris = load_iris()
  data = iris.data
  target = iris.target

  # Convert to PyTorch tensors
  data_tensor = torch.tensor(data, dtype=torch.float32)
  target_tensor = torch.tensor(target, dtype=torch.long)

  # Split dataset into train and test sets
  train_data, test_data, train_target, test_target = train_test_split(data_tensor, target_tensor, test_size=0.2,
                                                                      random_state=42)

  # Create data loaders (optional, if you want to batch and shuffle the data)
  train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_target), batch_size=1,
                                             shuffle=True)
  test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_target), batch_size=1,
                                            shuffle=False)

  train_inputs = torch.empty(0, 4, device=device)
  train_labels = torch.empty(0, dtype=torch.long, device=device)
  test_inputs = torch.empty(0, 4, device=device)
  test_labels = torch.empty(0, dtype=torch.long, device=device)

  # Concatenate all data into a single tensor on the specified device
  for data, labels in train_loader:
    train_inputs = torch.cat((train_inputs, data.to(device)), dim=0)
    train_labels = torch.cat((train_labels, labels.to(device)), dim=0)

  for data, labels in test_loader:
    test_inputs = torch.cat((test_inputs, data.to(device)), dim=0)
    test_labels = torch.cat((test_labels, labels.to(device)), dim=0)

  dataset = {}
  dataset['train_input'] = train_inputs
  dataset['test_input'] = test_inputs
  dataset['train_label'] = train_labels
  dataset['test_label'] = test_labels

  return dataset


iris_dataset = load_iris_dataset()

model = KAN(width=[4, 5, 3], grid=5, k=3, seed=0, device=device)

model(iris_dataset['train_input'])
model.plot(beta=100, scale=1, in_vars=['SL', 'SW', 'PL', 'PW'], out_vars=['Set', 'Ver', 'Vir'])


def train_acc():
  return torch.mean((torch.argmax(model(iris_dataset['train_input']), dim=1) == iris_dataset['train_label']).float())


def test_acc():
  return torch.mean((torch.argmax(model(iris_dataset['test_input']), dim=1) == iris_dataset['test_label']).float())

image_folder = 'video_img'
results = model.fit(iris_dataset, opt="Adam", metrics=(train_acc, test_acc),
                    loss_fn=torch.nn.CrossEntropyLoss(), steps=100, lamb=0.01, lamb_entropy=10., save_fig=True,
                    img_folder=image_folder)

print(results['train_acc'][-1], results['test_acc'][-1])
model.plot(scale=1, in_vars=['SL', 'SW', 'PL', 'PW'], out_vars=['Set', 'Ver', 'Vir'])


video_name='video'
fps=10

fps = fps
files = os.listdir(image_folder)
train_index = []
for file in files:
    if file[0].isdigit() and file.endswith('.jpg'):
        train_index.append(int(file[:-4]))

train_index = np.sort(train_index)

image_files = [image_folder+'/'+str(train_index[index])+'.jpg' for index in train_index]

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(video_name+'.mp4')