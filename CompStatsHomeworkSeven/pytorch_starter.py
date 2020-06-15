import torch as tc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
from matplotlib import pyplot as plt

# set parameters for neural network
"""test around with these settings a bit and see how the training changes.."""
learning_rate = 0.005  # learning rate for optimizer
n_epochs = 4000  # number of training epochs
batch_size = 128  # SGD minibatch size

# load data from csv file
data = []
with open('sheet7.csv') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for d in csvreader:
        data.append(d)
data.pop(0)  # first row in csv file is "x","y"
data = [[float(x), float(y)] for (x, y) in data]
data = tc.tensor(data)
X = data[:, 0:1]
Y = data[:, 1:2]


# create pytorch dataset
class Data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = (self.X[idx, :], self.Y[idx, :])
        return sample, idx


dataset = Data(X, Y)
dataloader = DataLoader(dataset, batch_size)


# define neural network
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(NeuralNet, self).__init__()
        """define your neural network architecture components"""
        """in the style: self.layer = nn.Linear..."""

    def forward(self, x):
        """define functional sequence of components"""
        """in the style: x = self.layer(x)"""
        return x


"""define other network classes """

# initialize model
"""construct instances of networks"""
"""e.g. model1 = NeuralNet(..)"""


def train(model, n_epochs):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(n_epochs):
        for i, (batch, idx) in enumerate(dataloader):
            inp, tgt = batch  # load input and target
            optimizer.zero_grad()  # set gradients to zero
            pred = model.forward(inp)  # predict with model
            loss = criterion(pred, tgt)
            loss.backward()  # backpropagation
            optimizer.step()  # gradient step
        if epoch % (n_epochs / 10) == 0:
            print("epoch {}, loss: {}".format(epoch, loss.item()))
    return model


# train your nets
"""e.g. model1 = train(model1, n_epochs)"""

# plot results
x = tc.squeeze(X).numpy()
y = tc.squeeze(Y).numpy()

# data
plt.plot(y, label="data")

# glm
""" define and estimate GLM and plot results"""

# neural net 1
# model1.eval()
""" plotting results of neural network 1"""

# neural net 2
# model2.eval()
""" plotting results of neural network 2"""

# neural net 3 (your own settings)
# model3.eval()
""" plotting results of neural network 2"""

plt.legend()
plt.show()
