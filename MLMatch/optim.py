import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import tqdm
from torch.utils.data import random_split, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.io import savemat, loadmat
from torch.autograd import Variable, grad
from scipy.fftpack import ifftn, fftshift
from numpy.random import normal

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
        dummy_input = torch.zeros(1, 1, 16, 16, 16) if pad else torch.zeros(1, 1, 10, 10, 10)
        self.flat_size = int(np.prod(self.conv_layers(dummy_input).size()[1:]))
        fc_layers = []
        in_features = self.flat_size
        for _ in range(depth_fc - 1):
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            in_features = out_features
        fc_layers.append(nn.Linear(in_features, 2))
        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        if self.pad:
            x = F.pad(x, (3, 3, 3, 3, 3, 3), "constant", 0)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)
    
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
    def __init__(self, depth_conv=1, depth_fc=2, out_channels = 8, out_features=32):
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
        dummy_input = torch.zeros(1, 1, 10, 10, 10)
        dummy_output = self.conv_layers(dummy_input)
        self.flat_size = int(np.prod(dummy_output.size()[1:]))
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## regressor % property scaler
depth_conv = 2
depth_fc = 8 
out_channels = 32 
out_features = 64 
model_dir = "your_regressor_path"  
model = CNNregressor(pad=False, depth_conv=depth_conv, depth_fc=depth_fc,
                          out_channels=out_channels, out_features=out_features).to(device)
model.load_state_dict(torch.load(os.path.join(model_dir, f"{your_regressor_fname}.pth")))
model.eval()

save_directory = "your_data_path"
data = loadmat(f"{save_directory}/data_regressor.mat")
s = data["s"]
z = data["z"]
c = data["c"]
y = np.stack((z.squeeze(), c.squeeze()), axis=1)
s_train, s_test, y_train, y_test = train_test_split(s, y, test_size=0.2, random_state=42)
batch_size = 32
y_scaler = StandardScaler()
y_train, y_test = y_scaler.fit_transform(y_train), y_scaler.transform(y_test)


# classifier model & data
depth_conv = 2
depth_fc = 2
out_channels = 8
out_features = 8
save_directory = "your_classifier_path"
state_feasible = torch.tensor(np.load(os.path.join(save_directory, "data_classifier_s_feasible.npy")))
state_infeasible = torch.tensor(np.load(os.path.join(save_directory, "data_classifier_s_infeasible.npy")))
dataset = FeasibilityDataset(state_feasible, state_infeasible)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
batch = next(iter(train_dataloader))

classifier = FeasibilityChecker(depth_conv=depth_conv, depth_fc=depth_fc, out_channels=out_channels, out_features=out_features).to(device)
classifier_save_path = f"{save_directory}/{your_classifier_fname}.pth"
classifier.load_state_dict(torch.load(classifier_save_path))


prob_threshold_item = 0.90
prob_threshold = torch.tensor(prob_threshold_item).unsqueeze(0).unsqueeze(0).to(device)


def z_scale(z_raw, scaler, device=device):
    return torch.tensor(scaler.transform([[z_raw, 0]])[0, 0], dtype=torch.float32)

def c_scale(c_raw, scaler, device=device):
    return torch.tensor(scaler.transform([[0, c_raw]])[0, 1], dtype=torch.float32)

def loss_constr(prediction, prob_threshold=prob_threshold):
    G = prob_threshold - prediction
    G_hat = torch.relu(G)  
    return G_hat

def loss_constr_amp(prediction, prob_threshold=prob_threshold):
    G = prob_threshold - prediction
    G_hat = torch.relu(G)  
    G_hat_amplified = torch.exp(G_hat) - 1
    return G_hat_amplified

def loss_obj(prediction, target, loss_type='l1'):
    if loss_type == 'l1':
        return F.l1_loss(prediction, target)
    elif loss_type == 'l2':
        return F.mse_loss(prediction, target)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
def compute_gradient(state, model, loss_fn, target, prop_idx, device=device, normalize=False):
    state = state.to(device).detach().requires_grad_(True)
    if prop_idx is not None:
        prediction = model(state)[0][prop_idx]
        loss = loss_fn(prediction, target.to(device))        
    else:
        prediction = model(state)        
        loss = loss_fn(prediction)        
    loss.backward()
    if normalize:
        grad_norm = state.grad.norm()
        return state.grad / (grad_norm + 1e-10), prediction.detach().cpu()
    else:
        return state.grad, prediction.detach().cpu()
    
def Heaviside(x, beta=5, eta=0.5):
    beta = torch.tensor(beta, dtype=x.dtype, device=x.device)
    eta = torch.tensor(eta, dtype=x.dtype, device=x.device)
    return (torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))) / (torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta)))

def periodic_check(s):
    periodic_sum = torch.sum((s[0,:,:] != s[-1,:,:])) + torch.sum((s[:,0,:] != s[:,-1,:])) + torch.sum((s[:,:,0] != s[:,:,-1]))
    return periodic_sum.item() == 0

def vf_compute(state):
    return torch.sum(state/10**3).item()    


z_target_scaled = z_scale(1.48e6, y_scaler)
z_target_item = z_target_scaled.detach().cpu().squeeze().item()


def inv_optim(train_dataloader=train_dataloader, device=device,
              z_target_scaled=z_target_scaled,
              model=model, classifier=classifier, loss_obj=loss_obj):

    z_target_item = z_target_scaled.detach().cpu().squeeze().item()
    constraint_on = True
    iter_max = 250
    volume_move_limit = 0.01
    if constraint_on:
        lamda = 0.5
        lamda_max = 1
        lamda_list = np.linspace(lamda, lamda_max, iter_max)
    else:
        lamda = 0.00
        lamda_max = 0.00
        lamda_list = np.linspace(lamda, lamda_max, iter_max)      

    batch_idx = np.random.randint(len(train_dataloader))
    for i, (states, values) in enumerate(train_dataloader):
        if i == batch_idx:
            state = random.choice(states[values == 1]).to(device)
            break
    while len(state.shape) < 5:
        state = state.unsqueeze(0)

    zs, cs, states, vfs, feasible_prop = [], [], [], [], []
    states_init = []
    states_init.append(state.detach().cpu().numpy().squeeze())

    for i in range(iter_max):
        classifier.eval()
        model.eval()
        with torch.no_grad():
            z_pred, c_pred = model(state).detach().cpu().squeeze().numpy()
            p_prop = classifier(state).detach().cpu().squeeze().numpy()        
        zs.append(z_pred)
        cs.append(c_pred)            
        feasible_prop.append(p_prop)                    
        states.append(state.detach().cpu().numpy().squeeze())
        vfs.append(vf_compute(state.squeeze().detach().cpu()))

        grad_obj_z, z_pred = compute_gradient(
            state=state, model=model, loss_fn=loss_obj,
            target=z_target_scaled, prop_idx=0,
            device=device, normalize=False
        )
        grad_constr, prop_pred = compute_gradient(
            state=state, model=classifier, loss_fn=loss_constr,
            target=torch.tensor(1).unsqueeze(0).unsqueeze(0).float().to(device),
            prop_idx=None, device=device, normalize=False
        )

        grad_obj = grad_obj_z                  
        norm_grad_obj = grad_obj.norm()
        norm_grad_constr = grad_constr.norm()
        grad_ratio = norm_grad_obj / norm_grad_constr if norm_grad_constr >= 1e-9 else 1

        state_cur = state.detach().clone()
        volume_change_checked = False
        grad_scale = 500
        lamda = lamda_list[i]

        grad_obj[:, :, [0, -1], :, :], grad_obj[:, :, :, [0, -1], :], grad_obj[:, :, :, :, [0, -1]] = 0, 0, 0
        grad_constr[:, :, [0, -1], :, :], grad_constr[:, :, :, [0, -1], :], grad_constr[:, :, :, :, [0, -1]] = 0, 0, 0

        while not volume_change_checked:
            update_step = grad_scale * (grad_obj + lamda * grad_ratio * grad_constr)
            state_new = state_cur - update_step.detach().clone()   
            state_new.clamp_(0, 1)
            state_new = Heaviside(state_new)
            state_new = (state_new >= 0.5).float()
            volume_change = vf_compute(torch.abs(state_new.squeeze() - state_cur.squeeze()))
            if volume_change >= volume_move_limit:
                grad_scale *= 0.95
            else:
                volume_change_checked = True
        state.data = state_new

    # final evaluation
    classifier.eval()
    model.eval()
    with torch.no_grad():
        z_pred, c_pred = model(state).detach().cpu().squeeze().numpy()
        p_prop = classifier(state).detach().cpu().squeeze().numpy()               

    zs.append(z_pred)
    cs.append(c_pred)            
    feasible_prop.append(p_prop)                    
    states.append(state.detach().cpu().numpy().squeeze())
    vfs.append(vf_compute(state.squeeze().detach().cpu()))

    return (
        np.array(zs),
        np.array(cs),
        np.array(states_init).squeeze(),
        np.array(vfs),
        np.array(feasible_prop)
    )


num_runs = 10
zs_reps, cs_reps, s_init_reps, vfs_reps, ps_reps = map(
    np.array, zip(*(inv_optim() for _ in range(num_runs)))
)