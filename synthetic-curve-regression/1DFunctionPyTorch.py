import torch
from torch import nn

import matplotlib.pyplot as plt

######################################
# Custom Dataset for 1D Function
######################################
class SimpleCurveData(torch.utils.data.Dataset):
  """
  Data loader for fitting a complex 1D function: y = max(2x, 3sin(5x) + x^2) over x ∈ [-2.5, 2.5].
  This dataset allows us to study neural network behavior and optimization without estimation error,
  as we can generate arbitrary amounts of training data from the known ground truth function.
  """

  def __init__(self):
    self.x_data = torch.arange(-2.5,2.5,0.001).unsqueeze(1)
    self.y_data = torch.max(2 * self.x_data, 3 * torch.sin(5 * self.x_data) + self.x_data ** 2)

  def __len__(self):
    return self.x_data.shape[0]

  def __getitem__(self, i):
    return self.x_data[i], self.y_data[i]

######################################
# Feed Forward Neural Network
######################################

class FeedForwardNetwork(nn.Module):

  def __init__(self, input_dim, layer_widths, output_dim, activation=nn.ReLU):
    super().__init__()
    self.layer_widths = layer_widths
    assert(len(layer_widths) >= 1)

    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(input_dim, layer_widths[0]))
    self.layers.append(activation())

    for i in range(1, len(layer_widths)):
      self.layers.append(activation())
      self.layers.append(nn.Linear(layer_widths[i - 1], layer_widths[i]))

    self.layers.append(nn.Linear(layer_widths[-1], output_dim))


  def forward(self, x):
    out = x

    for layer in self.layers:
      out = layer(out)

    return out

  def getParamCount(self):
    return sum(p.numel() for p in self.parameters())

######################################
# Training Loop
######################################

def train(model, dataloader):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
  loss_fn = torch.nn.MSELoss()

  for i in range(0, 100):
    for x, y in dataloader:
      out = model(x)
      loss = loss_fn(out, y)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    scheduler.step()
    # if i % 5 == 0:
    #   plotPredictions(model, dataloader, i + 1)
    
  plotPredictions(model, dataloader, "Final")


def plotPredictions(model, dataloader, epoch, save_fig=False):
  dataset = dataloader.dataset
  # Switch to eval mode
  model.eval()

  # Make predictions for the full dataset
  y_pred = model(dataset.x_data)

  # Compute loss for reporting
  loss = ((dataset.y_data-y_pred)**2).mean().item()

  # Plot dataset and predictions
  plt.figure(figsize=(10,4))
  plt.plot(dataset.x_data.data, dataset.y_data.data, color="black", linestyle='dashed', label="True Function")
  plt.plot(dataset.x_data.data, y_pred.data, color="red", label="Network")
  plt.text(-2.6,-2.3, "Epoch: {}".format(epoch))
  plt.text(-2.6,-3.0, "Loss: {:0.3f}".format(loss))
  plt.text(-2.6,-3.7, "Params: {}  {}".format(model.getParamCount(), str(model.layer_widths)))
  plt.ylim(-4,10)

  if save_fig:
    plt.savefig("epoch{}.png".format(epoch))
  else:
    plt.show()

  model.train()


if __name__ == "__main__":
    # Generate a dataset for our curve
    dataset = SimpleCurveData()

    # Set up a dataloader to grab batches
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)


    # Sanity Check: Visualize the whole dataset as a curve and a batch as points
    x_batch, y_batch = next(iter(dataloader))

    plt.figure(figsize=(10,4))
    plt.plot(dataset.x_data.data, dataset.y_data.data, color="black", linestyle='dashed', label="True Function")
    plt.scatter(x_batch, y_batch, c="red", s=40, label="Batch Samples",alpha=0.75, edgecolors='none')
    plt.legend()
    plt.title("Sanity Check: Curve Dataset and Dataloader")
    plt.show()


    ######################################
    # Experimentation
    ######################################

    # Build our network

    # Original network
    # layer_widths = [64]
    # model = FeedForwardNetwork(1, layer_widths, 1, activation=nn.ReLU)

    # Experimenting with depth, keeping # of parameters constant
    # w = 7
    # d = 4
    # layer_widths = [w] * d
    # model = FeedForwardNetwork(1, layer_widths, 1, activation=nn.ReLU)

    # Experimenting with activation functions
    # w = 7
    # d = 4
    # layer_widths = [w] * d
    # model = FeedForwardNetwork(1, layer_widths, 1, activation=nn.Mish)


    # To reach a loss of < 0.005, I:
    #   increased the learning rate
    #   added an exponential learning rate scheduler
    #   increased the width and depth of the network
    w = 16
    d = 8
    layer_widths = [w] * d
    model = FeedForwardNetwork(1, layer_widths, 1, activation=nn.Mish)


    train(model, dataloader)
