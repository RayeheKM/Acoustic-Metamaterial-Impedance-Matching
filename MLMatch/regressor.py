import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
from torch.optim import Adam
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class Data3D(Dataset):
    def __init__(self, s, y):
        self.s = torch.tensor(s, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)


class CNNregressor(nn.Module):
    def __init__(self, pad=True, depth_conv=2, depth_fc=2, out_channels=32, out_features=64):
        super(CNNregressor, self).__init__()
        self.pad = pad
        conv_layers = []
        in_channels = 1
        for _ in range(depth_conv):
            conv_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2
        self.conv_layers = nn.Sequential(*conv_layers)
        fc_layers = []
        in_features = self.flat_size
        for _ in range(depth_fc - 1):
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            in_features = out_features
        fc_layers.append(nn.Linear(in_features, 2))
        self.fc_layers = nn.Sequential(*fc_layers)
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_directory = "your_path"
data = loadmat(f"{save_directory}/data_regressor.mat")
s = data["s"]
z = data["z"]
c = data["c"]
y = np.stack((z.squeeze(), c.squeeze()), axis=1)

s_train, s_test, y_train, y_test = train_test_split(s, y, test_size=0.2, random_state=42)
batch_size = 32
y_scaler = StandardScaler()
y_train, y_test = y_scaler.fit_transform(y_train), y_scaler.transform(y_test)

train_loader = DataLoader(Data3D(s_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(Data3D(s_test, y_test), batch_size=batch_size, shuffle=False)

depth_conv = 2
depth_fc = 2
out_channels = 32
out_features = 64
model = CNNregressor(
    pad=False,
    depth_conv=depth_conv,
    depth_fc=depth_fc,
    out_channels=out_channels,
    out_features=out_features
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()
epochs = 250
train_losses = []
test_losses = []
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for states, targets in train_loader:
        states, targets = states.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(states).squeeze()
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for states, targets in test_loader:
            states, targets = states.to(device), targets.to(device)
            predictions = model(states).squeeze()
            test_loss += criterion(predictions, targets).item()
    test_loss /= len(test_loader)
    test_losses.append(test_loss)        