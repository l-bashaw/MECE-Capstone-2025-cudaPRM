import pytorch_kinematics as pk
import torch
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
urdf = "/home/lenman/capstone/parallelrm/resources/robot/stretch/stretch_spherized.urdf"

chain = pk.build_serial_chain_from_urdf(open(urdf, mode='rb').read(), "link_grasp_center", "base_link")
chain = chain.to(device=device, dtype=dtype)

N = 10000
th_batch = torch.randn(N, 8, device=device, dtype=dtype)

for i in range(3):     
    tg_batch = chain.forward_kinematics(th_batch)


start = time()
tg_batch = chain.forward_kinematics(th_batch)
end = time()
print("Batch forward kinematics time:", end - start)

# th = [0.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# ret = chain.forward_kinematics(th, end_only=False)
# tg = ret['link_grasp_center']
# m = tg.get_matrix()
# pos = m[:, :3, 3]
# rot = pk.matrix_to_quaternion(m[:, :3, :3])

# print("Position:", pos)
# print("Rotation:", rot)