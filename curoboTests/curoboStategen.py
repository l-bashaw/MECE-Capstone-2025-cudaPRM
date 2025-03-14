#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
""" Example computing collisions using curobo

"""
# Third Party
import torch
import time
import numpy as np
import pandas as pd

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

if __name__ == "__main__":
    robot_file = "ur5e.yml"
    world_file = "collision_table.yml"
    tensor_args = TensorDeviceType()
    config = RobotWorldConfig.load_from_config(
        robot_file, world_file, collision_activation_distance=0.05
    )
    curobo_fn = RobotWorld(config)


    num_iters = 10000
    batch_sizes = [100, 1000, 10000, 50000, 100000]
    valid = False
    times = np.zeros((num_iters, len(batch_sizes)))
    for batch_size in batch_sizes:
        q_batch = curobo_fn.sample(batch_size, mask_valid=valid)
        q_batch = curobo_fn.sample(batch_size, mask_valid=valid)
        for i in range(num_iters):
            torch.cuda.synchronize()
            start = time.time()
            q_batch = curobo_fn.sample(batch_size, mask_valid=valid)
            torch.cuda.synchronize()
            end = time.time()
            times[i, batch_sizes.index(batch_size)] = end - start

print(times.shape)

column_names = ['N=100', 'N=1000', 'N=10000', 'N=50000', 'N=100000']

# Convert the NumPy array to a pandas DataFrame with column names
df = pd.DataFrame(times, columns=column_names)

# Save the DataFrame to a CSV file
df.to_csv('curobo_Stategen_times.csv', index=False)