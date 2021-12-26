import torch
from utils import read_data, plot_data
import numpy as np
import matplotlib.pyplot as plt
import argparse


class ThreeLayerMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 10)
        self.layer2 = torch.nn.Linear(10, 50)
        self.output = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        out = torch.sigmoid(self.output(x))
        return out


class Dataset(torch.utils.data.Dataset):
    def __init__(self, xy_data, labels, device):
        self.xy_data = torch.tensor(xy_data, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)

        self.labels = self.labels.reshape(-1, 1)

    def __len__(self):
        return len(self.xy_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        preds = self.xy_data[idx, :]  # idx rows, all 4 cols
        lbl = self.labels[idx, :]    # idx rows, the 1 col
        sample = {'predictors': preds, 'target': lbl}
        return sample


def accuracy(model, ds):
    # ds is a iterable Dataset of Tensors
    n_correct = 0
    n_wrong = 0
    for i in range(len(ds)):
        inpts = ds[i]['predictors']
        ground_truth = ds[i]['target']    # float32  [0.0] or [1.0]
        with torch.no_grad():
            output = model(inpts)

        # avoid 'target == 1.0'
        if ground_truth < 0.5 and output < 0.5:  # .item() not needed
            n_correct += 1
        elif ground_truth >= 0.5 and output >= 0.5:
            n_correct += 1
        else:
            n_wrong += 1

    return (n_correct * 1.0) / (n_correct + n_wrong)


def predict(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    with torch.no_grad():
        predictions = model(x)
    predictions = torch.where(predictions > 0.5, 1, 0)  # pick a label
    return predictions.numpy()


def plot_decision_boundary(pred_func, points, labels, title):
    points = np.asarray(points)
    labels = np.asarray(labels)
    # Set min, max and generate a grid of points with distance hop between them
    x_min, x_max = points[:, 0].min() - .5, points[:, 0].max() + .5
    y_min, y_max = points[:, 1].min() - .5, points[:, 1].max() + .5
    hop = 0.01
    xv, yv = np.meshgrid(np.arange(x_min, x_max, hop),
                         np.arange(y_min, y_max, hop))
    # Predict the output for the whole grid
    Z = pred_func(np.c_[xv.ravel(), yv.ravel()])
    Z = Z.reshape(xv.shape)
    plt.title(title)
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.contourf(xv, yv, Z, cmap=plt.cm.Spectral)
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.binary)
    # plt.show()
    plt.savefig('-'.join(title.split()) + '.png')


def add_gauss_noise(point, sigma):
    noise_x = torch.tensor(np.random.normal(0, sigma, point.shape[0]),
                           dtype=torch.float32)
    noise_y = torch.tensor(np.random.normal(0, sigma, point.shape[0]),
                           dtype=torch.float32)
    point = point + torch.cat([noise_x, noise_y]).reshape(point.shape)
    return point


def train(jitter=False):
    for epoch in range(0, max_epochs):
        epoch_loss = 0.0            # for one full epoch

        for (_, batch) in enumerate(train_loader):
            XY = batch['predictors']  # [batch-size, 2]  inputs
            if jitter:
                XY = add_gauss_noise(XY, 0.1)
            targets = batch['target']      # [batch-size, 1]  targets
            output = model(XY)          # [batch-size, 1]

            loss_val = loss_func(output, targets)   # a tensor
            epoch_loss += loss_val.item()  # accumulate

            optimizer.zero_grad()  # reset all gradients
            loss_val.backward()   # compute all gradients
            optimizer.step()      # update all weights

        if epoch % log_interval == 0:
            print("epoch = %4d   loss = %0.4f" %
                  (epoch, epoch_loss))


parser = argparse.ArgumentParser(description='Train a three-layer model.')
parser.add_argument('--jitter', help='jitter input by adding noise',
                    action="store_true", default=False)
args = parser.parse_args()
print(args)

torch.manual_seed(1)
np.random.seed(1)

print("Creating a DataLoader ")
xy_data, labels = read_data()
plot_data(xy_data, labels)
device = torch.device("cpu")  # apply to Tensor or Module
train_ds = Dataset(xy_data, labels, device)

batch_size = 2
train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=batch_size, shuffle=True)

print("Creating 2-10-50-1 binary NN classifier ")
model = ThreeLayerMLP().to(device)

lrn_rate = 0.1
loss_func = torch.nn.BCELoss()  # binary cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr=lrn_rate)
max_epochs = 4500
log_interval = 100
print("Loss function: " + str(loss_func))
print("Optimizer: SGD")
print(f"Learn rate: {lrn_rate}")
print(f"Batch size: {batch_size}")
print(f"Max epochs: {max_epochs}")

print(f"Training for {max_epochs} epochs")
train(args.jitter)

model = model.train()  # set training mode
model = model.eval()
acc_train = accuracy(model, train_ds)  # we don't have a test set
print(f"Accuracy on train data = {(acc_train * 100):0.2f}")

title = "Noise Added to Smooth boundary" if args.jitter else "Known Overfit"
plot_decision_boundary(lambda x: predict(x),
                       xy_data, labels, title)
