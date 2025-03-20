import torch
import torchvision
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import datetime
import random
import string
import wandb


use_cuda_if_avail = True
use_mps_if_avail  = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
elif use_mps_if_avail and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

config = {
    "bs":2048,        # batch size
    "lr":0.003,       # learning rate
    "l2reg":0.00005,  # weight decay
    "lr_decay":0.99,  # exponential learning decay
    "aug":True,       # enable data augmentation
    "max_epoch":20
}

def generateRunName():
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  now = datetime.datetime.now()
  run_name = "["+random_string+"] MNIST  "+now.strftime("[%m-%d-%Y--%H:%M]")
  return run_name

wandb.init(
    project="MNIST CNN CS499 A3",
    name=generateRunName(),
    config=config
)


def main():

  # Get dataloaders
  train_loader, test_loader = getDataloaders(visualize=True)

  # Build model
  model = SimpleCNN()


  ###################################
  # Sanity Check
  ###################################
  x,y = next(iter(train_loader))
  out = model(x)
  assert(out.shape == (config["bs"], 10))
  
  # Start model training
  train(model, train_loader, test_loader)


###################################
# Data Augmentation
###################################
def getDataloaders(visualize = True):
  
  # Set up dataset and data loaders
  test_transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])

  if config["aug"]:
    train_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(15, (0.15, 0.15), (0.8, 1.2)),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
  else:
    train_transform = test_transform



  train_set = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
  test_set = datasets.MNIST('../data', train=False, download=True, transform=test_transform)

  train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=config["bs"])
  test_loader = torch.utils.data.DataLoader(test_set,  shuffle=False, batch_size=config["bs"])

  if visualize:
    # Plot out some transforms
    to_pil_image = transforms.ToPILImage()
    img = to_pil_image(train_set.data[0])

    fig, axs = plt.subplots(3,10, figsize=(10, 3))
    axs[0][0].imshow(img, cmap="grey")
    axs[0][0].get_xaxis().set_visible(False)
    axs[0][0].get_yaxis().set_visible(False)
    for i in range(1,30):
      axs[i//10][i%10].imshow(train_transform(img).squeeze(), cmap="grey")
      axs[i//10][i%10].get_xaxis().set_visible(False)
      axs[i//10][i%10].get_yaxis().set_visible(False)
    plt.show()

  return train_loader, test_loader


###################################
# Implement SimpleCNN
###################################

class SimpleCNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleList()

    # Block 1
    self.layers.append(nn.Conv2d(1, 36, 5))
    self.layers.append(nn.BatchNorm2d(36))
    # self.layers.append(nn.LayerNorm([36, 24, 24]))
    self.layers.append(nn.ReLU())
    self.layers.append(nn.Conv2d(36, 36, 5))
    self.layers.append(nn.BatchNorm2d(36))
    # self.layers.append(nn.LayerNorm([36, 20, 20]))
    self.layers.append(nn.ReLU())
    self.layers.append(nn.Conv2d(36, 36, 2, 2))
    # self.layers.append(nn.MaxPool2d(2))

    # Block 2
    self.layers.append(nn.Conv2d(36, 64, 3))
    self.layers.append(nn.BatchNorm2d(64))
    # self.layers.append(nn.LayerNorm([64, 8, 8]))
    self.layers.append(nn.ReLU())
    self.layers.append(nn.Conv2d(64, 128, 3))
    self.layers.append(nn.BatchNorm2d(128))
    # self.layers.append(nn.LayerNorm([128, 6, 6]))
    self.layers.append(nn.ReLU())
    self.layers.append(nn.Conv2d(128, 128, 2, 2))
    # self.layers.append(nn.MaxPool2d(2))

    # Block 3
    self.layers.append(nn.Conv2d(128, 256, 1))
    self.layers.append(nn.BatchNorm2d(256))
    # self.layers.append(nn.LayerNorm([256, 3, 3]))
    self.layers.append(nn.ReLU())
    self.layers.append(nn.Conv2d(256, 10, 1))
    self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))

  def forward(self, x):
    out = x

    for layer in self.layers:
      out = layer(out)
    
    return out.squeeze()


###################################
# Compute Accuracy
###################################

def computeAccuracy(out, y):
  return torch.sum(torch.argmax(out, dim=1) == y) / y.size(dim=0)


############################################
# Training loop
############################################

def train(model, train_loader, test_loader):
  model.to(device)

  optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
  scheduler = ExponentialLR(optimizer, gamma=config["lr_decay"])

  criterion = nn.CrossEntropyLoss()

  iteration = 0
  for epoch in range(config["max_epoch"]):
    model.train()
   
    for x,y in train_loader:
      x = x.to(device)
      y = y.to(device)

      out = model(x)
      loss = criterion(out,y)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      acc = computeAccuracy(out, y)
      
      iteration+=1

      wandb.log({"Train Acc": acc, "Train Loss": loss}, step=iteration)


    # Evaluate on held out data
    test_loss, test_acc = evaluate(model, test_loader)

    wandb.log({"Test Acc": test_acc, "Test Loss": test_loss}, step=iteration)

    wandb.log({"Test Visualization": generatePredictionPlot(model, test_loader)}, step=iteration)

    scheduler.step()

  test_loss, test_acc = evaluate(model, test_loader, plot_misclassified=True)

  wandb.finish()



############################################
# Skeleton Code
############################################

def evaluate(model, test_loader, plot_misclassified=False):
  model.eval()

  running_loss = 0
  running_acc = 0
  criterion = torch.nn.CrossEntropyLoss(reduction="sum")
  misclassified = []
  b = 0
  for x,y in test_loader:

    x = x.to(device)
    y = y.to(device)

    out = model(x)
    loss = criterion(out,y)

    acc = computeAccuracy(out, y)*x.shape[0]

    running_loss += loss.item()
    running_acc += acc

    if plot_misclassified:
      misclassified.extend(getMisclassifiedIndices(out, y) + config["bs"] * b)
      b += 1

  if plot_misclassified:
    wandb.log({"Misclassification Visualization": generateMisclassifiedPlot(model, test_loader, misclassified)})

    

  return running_loss/len(test_loader.dataset), running_acc/len(test_loader.dataset)


def generatePredictionPlot(model, test_loader):
  model.eval()
  x,y = next(iter(test_loader))
  out = F.softmax(model(x.to(device)).detach(), dim=1)

  num = min(20, x.shape[0])
  f, axs = plt.subplots(2, num, figsize=(4*num,8))
  for i in range(0,num):
    axs[0,i].imshow(x[i,:,:].squeeze().cpu(), cmap='gray')
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)
    axs[1,i].bar(list(range(10)),out[i,:].squeeze().cpu(), label=list(range(10)))
    axs[1,i].set_xticks(list(range(10)))
    axs[1,i].set_ylim(0,1)

  return f

def getMisclassifiedIndices(out, y):
    predictions = torch.argmax(out, dim=1)

    incorrect_mask = predictions != y

    misclassified_indices = torch.where(incorrect_mask)[0]
    return misclassified_indices

def generateMisclassifiedPlot(model, test_loader, indices):
    model.eval()
    dataset = test_loader.dataset
    
    # Get the specific examples we want
    x = torch.stack([dataset[i][0] for i in indices])
    
    x = x.to(device)
    out = F.softmax(model(x.to(device)).detach(), dim=1)
    
    num = len(indices)    
    f, axs = plt.subplots(2, num, figsize=(4*num,8))
    for i in range(0,num):
      axs[0,i].imshow(x[i,:,:].squeeze().cpu(), cmap='gray')
      axs[0,i].get_xaxis().set_visible(False)
      axs[0,i].get_yaxis().set_visible(False)
      axs[1,i].bar(list(range(10)),out[i,:].squeeze().cpu(), label=list(range(10)))
      axs[1,i].set_xticks(list(range(10)))
      axs[1,i].set_ylim(0,1)

    return f


if __name__ == "__main__":
  main()
