import torch
import torchvision
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import math
import umap.umap_ as umap
import os
import pickle
import pickle5
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.multinomial import Multinomial
from torchvision.datasets import FashionMNIST, CIFAR100,ImageFolder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from tqdm import tqdm
import gc
from torch.utils.data import Dataset
import wandb
import time
from torch.utils.tensorboard import SummaryWriter
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the root directory where the ImageNet dataset is stored
DATASET_DIR = '/D/datasets/imagenet/'

# transform = transforms.Compose([transforms.Resize((299, 299)),
#                                 transforms.ToTensor()])

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ImageNet dataset and select the first 100,000 samples
train_dataset = datasets.ImageNet(root=DATASET_DIR, split='train', transform=transform)
train_subset_dataset = torch.utils.data.Subset(train_dataset, range(100000))

val_dataset = datasets.ImageNet(root=DATASET_DIR, split='val', transform=transform)
val_subset_dataset = torch.utils.data.Subset(val_dataset, range(20000))

# Create a data loader to iterate over the subset dataset
# train_data_loader = torch.utils.data.DataLoader(train_subset_dataset, batch_size=32, shuffle=True)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_data_loader = torch.utils.data.DataLoader(val_subset_dataset, batch_size=32, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

# TEST_DATASET_DIR = '/D/datasets/imagenet/test'
# test_dataset = ImageFolder(TEST_DATASET_DIR, transform=transform)
# test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# for x, y in test_data_loader:
#     break
# plt.imshow(np.transpose(x[0], (1, 2, 0))
# print(y[0])

# Load the pretrained Inception V3 model
model = torchvision.models.inception_v3(pretrained=True)
# _ = model.eval()

# Move the model to the GPU if available
_ = model.to(device)

def test_model(model, test_data_loader):
    model.eval()
    total_correct = 0
    total_incorrect = 0
    with torch.no_grad():
        for x, y in tqdm(test_data_loader):
            output = model(x.to(device))
            if type(output) == tuple:
                logits = output[-1].squeeze(0)
            else:
                logits = output
            predictions = torch.argmax(torch.softmax(logits, dim=-1), dim=1)#.to(torch.device('cpu'))
            correct_classifications = sum(predictions == y.to(device))
            incorrect_classifications = len(x) - correct_classifications
            total_correct += correct_classifications
            total_incorrect += incorrect_classifications
    model.train()
    print(f"acc: {print(total_correct / (total_correct + total_incorrect))}")

# test_model(model, val_data_loader)

# Replace the last layer with an identity layer
model.fc = torch.nn.Identity()

class LogitsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
    
def get_logits_dataloader(model, original_loader, batch_size=32, whiten=False):
    logits_data_list = []
    logits_labels_list = []
    with torch.no_grad():
        for x, y in tqdm(original_loader):
            logits = model(x.to(device))
            logits_data_list.append(logits.to(torch.device('cpu')))
            logits_labels_list.append(y.to(torch.device('cpu')))

    logits_data_set = LogitsDataset(torch.concat(logits_data_list), torch.concat(logits_labels_list))
    logits_dataloader = DataLoader(logits_data_set, batch_size=batch_size, shuffle=True)
    if whiten:
        # Apply whitening to the features
        scaler = StandardScaler()
        outputs = scaler.fit_transform(torch.concat(logits_data_list).numpy())
        pca = PCA()
        whitened = pca.fit_transform(outputs)

        whitened_logits_data_set = LogitsDataset(whitened, torch.concat(logits_labels_list))
        whitened_logits_dataloader = DataLoader(whitened_logits_data_set, batch_size=batch_size, shuffle=True)
        return whitened_logits_dataloader
    else:
        return logits_dataloader
    
LOGITS_TRAIN_DATALOADER_PATH = '/D/datasets/imagenet/logits_dataloaders/logits_trian_dataloader.pkl'
LOGITS_VAL_DATALOADER_PATH = '/D/datasets/imagenet/logits_dataloaders/logits_val_dataloader.pkl'

# Load
with open(LOGITS_TRAIN_DATALOADER_PATH, 'rb') as f:
    logits_trian_dataloader = pickle5.load(f)
with open(LOGITS_VAL_DATALOADER_PATH, 'rb') as f:
    logits_val_dataloader = pickle5.load(f)

EPSILON = 1e-15

def get_multivariate_gaussian_entropy(std, epsilon=EPSILON):
    """
    $$H(N_D(\mu,\Sigma))=\frac{D}{2}(1+log(2\pi))+\frac{1}{2}log|\Sigma|$$
    Assuming a diagonal cov matrix represented as a vector
    Casting to numpy as it can compute log of smaller numbers
    """
    std_array = std.to('cpu').detach().numpy().astype(np.float128)  # no need to use float128
    if not (std_array > 0).all():
        raise ValueError('Got a non positive entry in diagonal cov matrix')
    return np.maximum(epsilon, ((std_array.shape[-1] / 2) * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.prod(std_array, axis=-1)))).mean()


def get_multinomial_entropy(logits, epsilon=EPSILON):
    """
    Receives unactivates logits
    epsilon replaces 0 probability that results from torch's low float resolution
    """
    probs = logits.detach().squeeze(0).softmax(-1)
    probs = probs.to('cpu').numpy().astype(np.float128)
    probs = np.clip(probs, epsilon, 1 - epsilon)
    return ((probs) * np.log(1/probs)).sum(-1).mean()


def get_kl_between_gaussians(mu1, std1, mu2, std2):
    pass

def reparametrize(mu, std, device):
    """
    Performs reparameterization trick z = mu + epsilon * std
    Where epsilon~N(0,1)
    """
    mu = mu.expand(1, *mu.size())
    std = std.expand(1, *std.size())
    eps = torch.normal(0, 1, size=std.size()).to(device)
    return mu + eps * std

class VIB(nn.Module):
    """
    Classifier with stochastic layer and KL regularization
    """
    def __init__(self, hidden_size, device):
        super(VIB, self).__init__()
        self.device = device
        self.description = 'Vanilla IB VAE as per the paper'
        self.hidden_size = hidden_size
        self.k = hidden_size // 2
        self.train_loss = []
        self.test_loss = []

        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(self.k, 1000)

        # Xavier initialization
        for _, module in self._modules.items():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                        module.bias.data.zero_()
                        continue
            for layer in module:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                            layer.bias.data.zero_()
        
        # Init with a small bias in std to create initial high entropy
        biased_layer = self.encoder[2]
        slice_size = biased_layer.bias.shape[0] // 2
        new_bias = torch.concat((biased_layer.bias[:slice_size], torch.ones(slice_size) * 0.1))
        biased_layer.bias.data = new_bias


    def forward(self, x):
        z_params = self.encoder(x)
        mu = z_params[:, :self.k]
        # softplus transformation (soft relu) and a -5 bias is added as in the paper
        # std = F.softplus(z_params[:, self.k:] - 5, beta=1)
        # std = F.softplus(z_params[:, self.k:] - 1, beta=1)
        std = F.softplus(z_params[:, self.k:], beta=1)
        if self.training:
            z = reparametrize(mu, std, self.device)
        else:
            z = mu.clone().unsqueeze(0)
        n = Normal(mu, std)
        log_probs = n.log_prob(z.squeeze(0))  # These may be positive as this is a PDF
        
        logits = self.classifier(z)
        return (mu, std), log_probs, logits
   
def vib_loss(logits, labels, mu, std, beta):
    classification_loss = nn.CrossEntropyLoss()(logits.squeeze(0), labels)  # In torch cross entropy function applies the softmax itself
    normalization_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum()
    # KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return classification_loss + beta * normalization_loss


def loop_data(model, train_dataloader, test_dataloader, beta, model_save_path, epochs,
              device, optimizer=None, scheduler=None, eta=0.001,
             num_minibatches=1, is_ppo=False, run_name="test_run", is_dyn_beta=False):
    wandb.init(
    project='dynamic_beta',
    entity=None,
    sync_tensorboard=True,
    config=None,
    name=run_name,
    monitor_gym=False,
    save_code=True,)
    writer = SummaryWriter(f"runs/{run_name}")
    
    model.train()
    
    epoch_i_z_x_array = np.zeros(epochs)
    epoch_i_z_y_array = np.zeros(epochs)

    for e in tqdm(range(epochs)):
        epoch_loss = 0 
        epoch_classification_loss = 0
        epoch_total_kld = 0
        epoch_ratio1 = 0
        epoch_ratio2 = 0
        epoch_ratio3 = 0
        epoch_ratio4 = 0
        epoch_kld_from_base_new = 0
        epoch_kld_from_base_old = 0
        epoch_mean_y_entropy = 0
        epoch_mean_z_entropy = 0
        
        if is_dyn_beta:
            if e < 2:
                beta = EPSILON
            else:
                with torch.no_grad():
                    epoch_i_z_x_delta = epoch_i_z_x_array[e - 1] - epoch_i_z_x_array[e - 2]
                    epoch_i_z_y_delta = epoch_i_z_y_array[e - 1] - epoch_i_z_y_array[e - 2] 
                    beta = epoch_i_z_y_delta / epoch_i_z_x_delta
                    if (epoch_i_z_x_delta == 0) or (beta > 1):
                        beta = EPSILON
                    
        for batch_num, (embeddings, labels) in enumerate(train_dataloader):
            # Compute base z distribution
            with torch.no_grad():
                x = embeddings.to(device)
                (base_mu, base_std), base_log_probs, _ = model(x)
                
            for i in range(num_minibatches):
                x = embeddings.to(device)
                y = labels.to(device)
                (mu, std), log_probs, logits = model(x)
#                 activated = torch.softmax(logits, -1)
                
                if np.isnan(get_multinomial_entropy(logits)):
                    pdb.set_trace()
    
                with torch.no_grad():
                    epoch_i_z_x_array[e] += get_multivariate_gaussian_entropy(std) / len(train_dataloader)
                    epoch_i_z_y_array[e] += get_multinomial_entropy(logits) / len(train_dataloader)
    
                kld_from_std_normal = (-0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2))).sum(1).mean(0, True)
                log_ratio = log_probs - base_log_probs
            
                if is_ppo:
                    with torch.no_grad():
                        kld_from_base_dist_old = torch.distributions.kl_divergence(
                            Normal(base_mu, base_std), Normal(mu, std)).sum(1).mean(0, True)
                        kld_from_base_dist = torch.distributions.kl_divergence(
                            Normal(mu, std), Normal(base_mu, base_std)).sum(1).mean(0, True)    

                        epoch_kld_from_base_new += kld_from_base_dist / num_minibatches
                        epoch_kld_from_base_old += kld_from_base_dist_old / num_minibatches

                        kld_from_base_dist = kld_from_base_dist_old + kld_from_base_dist

                classification_loss = nn.CrossEntropyLoss()(logits.squeeze(0), y)
                
                if is_ppo:
                    # Replacing Beta
                    ratio = kld_from_base_dist / (kld_from_std_normal + kld_from_base_dist)
                    minibatch_loss = classification_loss + (beta * ratio).mean() * kld_from_std_normal.sum()
                else:    
                    minibatch_loss = classification_loss + beta * kld_from_std_normal

                optimizer.zero_grad()
                minibatch_loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    epoch_total_kld += kld_from_std_normal / num_minibatches
                    epoch_classification_loss += classification_loss.item() / num_minibatches
                    if is_ppo:
                        if i == 0:
                            epoch_ratio1 += ratio
                        elif i == 1:
                            epoch_ratio2 += ratio
                        elif i == 2:
                            epoch_ratio3 += ratio
                        elif i == 3:
                            epoch_ratio4 += ratio

            epoch_loss += minibatch_loss.item()
        
        epoch_loss /= batch_num
        model.train_loss.append(epoch_loss)
        writer.add_scalar("charts/epoch_train_loss", epoch_loss, e)
        writer.add_scalar(
            "charts/epoch_classification_loss", epoch_classification_loss / len(train_dataloader), e)
        writer.add_scalar(
            "charts/epoch_total_kld", epoch_total_kld / len(train_dataloader), e)

        if is_dyn_beta:
            writer.add_scalar("charts/epoch_i_z_x", epoch_i_z_x_array[e], e)
            writer.add_scalar("charts/epoch_i_z_y", epoch_i_z_y_array[e], e)
            writer.add_scalar("charts/epoch_dyn_beta", beta, e)

        if is_ppo:
            writer.add_scalar(
                "charts/epoch_ratio1", epoch_ratio1 / len(train_dataloader), e)
            writer.add_scalar(
                "charts/epoch_ratio2", epoch_ratio2 / len(train_dataloader), e)
            writer.add_scalar(
                "charts/epoch_ratio3", epoch_ratio3 / len(train_dataloader), e)
            writer.add_scalar(
                "charts/epoch_ratio4", epoch_ratio4 / len(train_dataloader), e)
            writer.add_scalar(
                "charts/avg_epoch_ratio", (epoch_ratio1 + epoch_ratio2 +
                                           epoch_ratio3 + epoch_ratio4) / (len(train_dataloader) * 4), e)
            writer.add_scalar(
                "charts/total_epoch_ratio", (epoch_ratio1 + epoch_ratio2 + epoch_ratio3 + epoch_ratio4) / len(train_dataloader), e)
            writer.add_scalar(
                "charts/epoch_kld_from_base_old", epoch_kld_from_base_old / len(train_dataloader), e)
            writer.add_scalar(
                "charts/epoch_kld_from_base_new", epoch_kld_from_base_new / len(train_dataloader), e)
            
        # test loss
        model.eval()
        epoch_test_loss = 0
        for batch_num, (embeddings, labels) in enumerate(test_dataloader):
            x = embeddings.to(device)
            y = labels.to(device)
            (mu, std), log_probs, logits = model(x)
            epoch_test_loss += vib_loss(logits, y, mu, std, beta).item()
            # TODO: add this line
            # epoch_test_loss += vib_loss(logits, y, mu, std, beta).item() / len(train_dataloader)
        model.test_loss.append(epoch_test_loss)
        writer.add_scalar("charts/epoch_val_loss", epoch_test_loss, e)
        model.train()

LR = 1e-4
EPOCHS = 70
BETA = 1
NUM_MINIBATCHES = 1

run_name = f"vib_imagenet_beta_new_dyn_beta_time_{int(time.time())}"

new_dyn_imagenet_vib_classifier = VIB(2048, device).to(device)
optimizer = optim.Adam(new_dyn_imagenet_vib_classifier.parameters(), LR / NUM_MINIBATCHES, betas=(0.5,0.999))
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

loop_data(new_dyn_imagenet_vib_classifier, logits_trian_dataloader, logits_val_dataloader, BETA,
          num_minibatches=NUM_MINIBATCHES, model_save_path='/tmp', epochs=EPOCHS, device=device,
          optimizer=optimizer, scheduler=scheduler, is_ppo=False, run_name=run_name, is_dyn_beta=True)

test_model(new_dyn_imagenet_vib_classifier, logits_val_dataloader)