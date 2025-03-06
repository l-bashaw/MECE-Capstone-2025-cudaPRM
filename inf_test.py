from inference import InferenceEngine
import numpy as np


model_state_path = "/home/lenman/capstone/parallelrm/models/percscore-nov12-50k.pt"
batch_size = 1

engine = InferenceEngine(batch_size, model_state_path)

# Create a random input tensor
input_tensor = np.random.rand(batch_size, 7+engine.pt_model.label_size).astype(np.float32)

engine.allocate_buffers()

engine.infer(input_tensor)
