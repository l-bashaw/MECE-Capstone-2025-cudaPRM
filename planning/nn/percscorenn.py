import pytorch_lightning as pl
import torch
import torch.nn as nn

# label_map = {
#   "human": [1, 0, 0],
#   "monitor": [0, 1, 0],
#   "cup": [0, 0, 1]
# }


class PercScoreProxyNet(pl.LightningModule):

  def __init__(self, label_size):
    super().__init__()
    self.label_size = label_size  
    num_features = 7 + label_size # 7 features for the object pose
    self.layers = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 256),
        nn.ReLU(), # add dropout
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    self.ce = nn.MSELoss(reduction='mean')

  def forward(self, x):
    # print("x.shape ", x.shape)
    # print("layersx.shape ", self.layers(x).shape)
    return self.layers(x).squeeze()
  
  def warmup(self):
    with torch.no_grad():
      self.forward(torch.rand(1, 7+self.label_size, device = self.device))