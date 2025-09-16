import random
import os
import torch
from torch.optim import Adam
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from torch.utils.data import random_split, Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from scipy.fftpack import ifftn, fftshift
from numpy.random import normal
import logging
import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def generate_3D_GRF(size, scale=1.0, sigma=1.0): ## (Sec. 2.7.2; Alg. 7 in the Appendix)
    kx, ky, kz = np.mgrid[:size[0], :size[1], :size[2]]
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    k[0, 0, 0] = 1
    power_spectrum = np.exp(-scale * k**2 / 2.0)
    phases = np.exp(2j * np.pi * np.random.rand(*size))
    fourier_space = power_spectrum * phases
    grf = np.real(ifftn(fourier_space))
    grf = (grf - np.mean(grf)) / np.std(grf) * sigma
    return grf

class FeasibilityDataset(Dataset):
    def __init__(self, state_feasible, state_infeasible):
        self.feasibility_data = [(state, 1) for state in state_feasible] + \
                                [(state, 0) for state in state_infeasible]
    def __len__(self):
        return len(self.feasibility_data)
    def __getitem__(self, idx):
        state, label = self.feasibility_data[idx]
        state_tensor = state.unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return state_tensor, label_tensor
    
class FeasibilityChecker(nn.Module):
    def __init__(self, depth_conv=1, depth_fc=2, out_channels=8, out_features=32):
        super(FeasibilityChecker, self).__init__()
        conv_layers = []
        in_channels = 1
        self.out_channels = out_channels
        self.out_features = out_features
        for _ in range(depth_conv):
            conv_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2
        self.conv_layers = nn.Sequential(*conv_layers)
        fc_layers = []
        in_features = self.flat_size
        out_features = self.out_features
        for _ in range(depth_fc - 1):
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            in_features = out_features
        fc_layers.append(nn.Linear(in_features, 1))
        fc_layers.append(nn.Sigmoid())
        self.fc_layers = nn.Sequential(*fc_layers)
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    

def augment_data(states):
    states_augmented = torch.cat([states, states.flip(2), states.flip(3)], dim=0)
    for i in range(1, 4):
        states_augmented = torch.cat([states_augmented, states.rot90(i, dims=(2, 3))], dim=0)
    return torch.cat([states, states_augmented], dim=0)    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.set_printoptions(formatter={'all': lambda x: f"{x:0.2f}"})
logging.basicConfig(filename='matlab_errors.log', level=logging.ERROR)    
       
depth_conv = 2
depth_fc = 2
out_channels = 8
out_features = 8

save_directory = "your_path"
state_feasible = torch.tensor(np.load(os.path.join(save_directory, "data_classifier_s_feasible.npy")))
state_infeasible = torch.tensor(np.load(os.path.join(save_directory, "data_classifier_s_infeasible.npy")))
dataset = FeasibilityDataset(state_feasible, state_infeasible)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
batch = next(iter(train_dataloader))
classifier = FeasibilityChecker(depth_conv=depth_conv, depth_fc=depth_fc, out_channels=out_channels, out_features=out_features).to(device)
total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
data_augmentation_during_training = True


model = FeasibilityChecker(
    depth_conv=depth_conv,
    depth_fc=depth_fc,
    out_channels=out_channels,
    out_features=out_features
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.BCELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.90)
epochs = 400
print_interval = epochs // 10 if epochs >= 10 else 1

train_losses = []
test_losses = []
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        states, values = batch
        states, values = states.to(device), values.to(device)
        len_state_origin = len(states)
        if 'data_augmentation_during_training' in globals() and data_augmentation_during_training:
            if 'augment_data' in globals():
                states = augment_data(states)
                factor = states.shape[0] // len_state_origin
                values = values.repeat(factor)
        optimizer.zero_grad()
        predicted_values = model(states).squeeze()
        if len(predicted_values.shape) == 0:
            predicted_values = predicted_values.unsqueeze(0)
        loss = criterion(predicted_values, values.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= max(1, len(train_dataloader))
    train_losses.append(train_loss)
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            states, values = batch
            states, values = states.to(device), values.to(device)
            predicted_values = model(states).squeeze()
            if len(predicted_values.shape) == 0:
                predicted_values = predicted_values.unsqueeze(0)
            loss = criterion(predicted_values, values.float())
            test_loss += loss.item()
    test_loss /= max(1, len(test_dataloader))
    test_losses.append(test_loss)
    if epoch % print_interval == 0 or epoch == epochs - 1:
        print(f"[Epoch {epoch+1}/{epochs}] train_loss={train_loss:.6f}  test_loss={test_loss:.6f}")
    scheduler.step()