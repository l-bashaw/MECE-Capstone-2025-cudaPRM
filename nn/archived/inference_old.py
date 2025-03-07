# Source: https://medium.com/@zergtant/accelerating-model-inference-with-tensorrt-tips-and-best-practices-for-pytorch-users-7cd4c30c97bc

# This file contains the TensorRT wrapper and utilities 
# for the neural surrogate model used in the PerceptionPRMPlanner.

# TensorRT enables accelerated inference on NVIDIA GPUs.
import torch
import tensorrt as trt
import onnx
import subprocess
from percscorenn import PercScoreProxyNet
import os
import torch2trt

import numpy as np



class InferenceEngine:
    def __init__(self, batch_size, model_path, onnx_path=None):
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            raise ValueError("CUDA not available. TensorRT only supports GPU acceleration.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File: {model_path} not found.")
        if model_path.endswith(".pt"):
            self.model_path = model_path
        else:
            raise ValueError("Model file must be a .pt file.")
        if onnx_path is not None:
            if onnx_path.endswith(".onnx"):
                self.onnx_path = onnx_path
            else:
                raise ValueError("ONNX file must be a .onnx file.")
        else:
            self.onnx_path = model_path.replace(".pt", ".onnx")  # If no ONNX path is provided, use the same path as the .pt model file with an .onnx extension.

        self.batch_size = batch_size
        
        
        def load_model(model_path) -> PercScoreProxyNet: 
            '''
            Load a PyTorch model from a .pt file.

            Args:
                model_path (str): Path to the model (.pt) file.
            '''
            state_dict = torch.load(model_path)
            label_size = state_dict['layers.0.weight'].shape[1] - 7 # 7 features for the object pose
            pt_model = PercScoreProxyNet(label_size=label_size)
            pt_model.load_state_dict(state_dict)
            pt_model.eval()
            return pt_model
        
        def to_onnx(pt_model, onnx_path, verbose=False) -> onnx.ModelProto:
            '''
            Convert a PyTorch model to ONNX format.

            Args:
                pt_model (PercScoreProxyNet): PyTorch model.
                onnx_path (str): Path where the ONNX model will be saved.
                verbose (bool): Whether to print onnx export process details.
            '''
            dummy_input = torch.randn(1, 7 + pt_model.label_size) 
            torch.onnx.export(pt_model, dummy_input, onnx_path, verbose=verbose)
            model_onnx = onnx.load(onnx_path)
            return model_onnx
        
        def build_engine(onnx_model, pt_model) -> trt.ICudaEngine:
            ''' 
            Build a TensorRT engine from an ONNX model.
            I do not fully understand this code.
            
            Args:
                model_onnx (onnx.ModelProto): ONNX model.
            '''
            LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(LOGGER)  # Create a TensorRT builder and network
            net = builder.create_network(0)
            config = builder.create_builder_config()    
            runtime = trt.Runtime(LOGGER)

            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
            populate_network(net, weights)

            parser = trt.OnnxParser(net, builder.logger)
            parser.parse(onnx_model.SerializeToString())

            profile = builder.create_optimization_profile()   # Set the optimization profile and builder params

            # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/OptimizationProfile.html
            profile.set_shape("input", (1, 7 + pt_model.label_size), (self.batch_size, 7 + pt_model.label_size), (self.batch_size*2, 7 + pt_model.label_size))  # Min, Opt, Max batch size = 1, 1000, 10000, TODO: Set this intelligently  
            
            #builder_config.max_workspace_size_bytes = 1 << 30  # 1 GB max workspace size  https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Engine.html
            #builder_config.flags = 1 << int(trt.BuilderFlag.STRICT_TYPES) # Strict type constraints

            engine = builder.build_engine(net, builder_config) 
            return engine, builder

        self.pt_model = load_model(model_path)    
        self.onnx_model = to_onnx(self.pt_model, self.onnx_path)
        self.engine, self.builder = build_engine(self.onnx_model, self.pt_model)

    def allocate_buffers(self, input_on_device = True):
        '''
        Allocate memory for the input and output tensors on the GPU.

        '''
        
        input_shape = (self.batch_size, 7 + self.pt_model.label_size)
        output_shape = (self.batch_size, 1)
        input_buf = trt.cuda.alloc_buffer(self.builder.max_batch_size * trt.volume(input_shape) * trt.float32.itemsize)
        output_buf = trt.cuda.alloc_buffer(self.builder.max_batch_size * trt.volume(output_shape) * trt.float32.itemsize)
        return input_buf, output_buf
       

    def infer(self, input_data: np.ndarray) -> torch.Tensor:
        '''
        Perform inference on the input data. Input is assumed to be on the CPU.

        Args:   
            input_data (torch.Tensor): Input data tensor on the CPU.
        '''
        if input_data.shape[0] != self.batch_size:
            raise ValueError(f"Input data batch size ({input_data.shape[0]}) does not match the engine batch size ({self.batch_size}).")
        if input_data.device != self.device:
            print(f"Moving input data to ({self.device}).\nInput data currently on: {input_data.device}")
            input_data = input_data.to(self.device)
        

        input_buf, output_buf = self.allocate_buffers()
        context = self.engine.create_execution_context()

        output_data = np.empty((self.batch_size, 1), dtype=np.float32)
        input_buf.host = input_data.ravel()
        trt_outputs = [output_buf.device]
        trt_inputs = [input_buf.device]

        # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html#tensorrt.IExecutionContext
        context.execute_async_v2(bindings=trt_inputs + trt_outputs, stream_handle=trt.cuda.Stream())
        output_buf.device_to_host()
        output_data[:] = np.reshape(output_buf.host, (self.batch_size, 1))
        
        print(output_data)