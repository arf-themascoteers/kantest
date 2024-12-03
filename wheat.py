from kan import *
from kan.utils import ex_round
import matplotlib
#matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[100,20,1], grid=5, k=3, seed=42, device=device)

df = pd.read_csv("wheat.csv").to_numpy()
df_train, df_test = train_test_split(df, train_size=0.9)

train_input = torch.tensor(df_train[:,0:-1], dtype=torch.float64)
train_label = torch.tensor(df_train[:,-1:], dtype=torch.float64)
test_input = torch.tensor(df_test[:,0:-1], dtype=torch.float64)
test_label = torch.tensor(df_test[:,-1:], dtype=torch.float64)

dataset = {
    "train_input" : train_input,
    "train_label" : train_label,
    "test_input": test_input,
    "test_label": test_label
}

#dataset = create_dataset(f, n_var=2, device=device)

print(dataset['train_input'].shape, dataset['train_label'].shape)



model(dataset['train_input'])
# model.plot()
# plt.show()
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
# model.plot()
# plt.show()
model = model.prune()
# model.plot()
# plt.show()

y_pred = model(dataset['test_input'])
r2 = r2_score(dataset['test_label'].cpu().detach().numpy(), y_pred.cpu().detach().numpy())
print(r2)