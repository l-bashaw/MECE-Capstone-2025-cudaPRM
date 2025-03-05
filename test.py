import numpy as np
import torch
import time

start = time.time()
x = torch.randn(10000, 6)
torch.cuda.synchronize()  # Make sure initialization is complete
start1 = time.time()
x = torch.randn(10000, 6, device='cuda', dtype=torch.float32)
end = time.time()
print(f"Time taken for randoms: {end - start}\nTime taken for randoms1: {end - start1}")

