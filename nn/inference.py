from torch2trt import torch2trt
import tensorrt as trt
from torch2trt import TRTModule
import torch
import percscorenn
import warnings

def build_pt_from_dict(model_path, label_size=3, to_cuda=True): 
    input_dim = torch.load(model_path)['layers.0.weight'].shape[1]
    if (input_dim - label_size) != 7:
        raise ValueError("Provided label size does not match the model's input dimension\nModel input dim: {}\nLabel size: {}".format(input_dim, label_size)+"\nHINT: input_dim = 7 + label_size")
    else:
        if to_cuda:
            model = percscorenn.PercScoreProxyNet(label_size).cuda()
        else:
            warnings.warn("Model is not loaded to cuda")
            model = percscorenn.PercScoreProxyNet(label_size)
        model.load_state_dict(torch.load(model_path))
        return model

def build_trt_from_pt(model, batch_size=10000, save_engine = False, file_name = None):
    x = torch.ones((batch_size, 7+3)).cuda()
    model_trt = torch2trt(model, [x], max_batch_size=batch_size)

    if save_engine:
        if file_name is None:
            torch.save(model_trt.state_dict(), "./model_trt.pth")
            print(f"Model saved to ./model_trt.pth")
        else:
            torch.save(model_trt.state_dict(), file_name)
            print(f"Model saved to {file_name}")

    return model_trt

def build_trt_from_dict(model_path, label_size = 3, batch_size=10000, to_cuda = True, save_engine = False, file_name = None):
    model = build_pt_from_dict(model_path, label_size, to_cuda)
    model_trt = build_trt_from_pt(model, batch_size, save_engine, file_name)
    return model_trt

def load_trt_engine(file_name):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(file_name))
    return model_trt


'''
model_state_path = "/home/lenman/capstone/parallelrm/models/percscore-nov12-50k.pt"
batch_size = 100
model_trt = build_trt_from_dict(model_state_path, batch_size=batch_size)
x = torch.rand((batch_size, 7+3)).cuda()
output = model_trt(x)
print(output)'
'''