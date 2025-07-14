import torch

from torch2trt import torch2trt
from nn.percscorenn import PercScoreProxyNet

class ModelLoader:
    def __init__(self, label_size=3, max_batch_size = 10000, use_trt=True):
        self.label_size = label_size
        self.max_batch_size = max_batch_size
        self.is_trt = use_trt

    def load_model(self, model_path):

        input_dim = torch.load(model_path)['layers.0.weight'].shape[1]
        
        if (input_dim - self.label_size) != 7:
            raise ValueError("Provided label size does not match the model's input dimension\nModel input dim: {}\nLabel size: {}".format(input_dim, self.label_size)+"\nHINT: input_dim = 7 + label_size")
        else:
            model = PercScoreProxyNet(self.label_size).cuda()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            if self.is_trt:
                x = torch.ones((self.max_batch_size, 7+self.label_size)).cuda()
                model_trt = torch2trt(model, [x], max_batch_size=self.max_batch_size)
                return model_trt
            else:
                return model
        


