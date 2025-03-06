import os
import random
import sys
import numpy as np
import common
import torch
import onnx
import common_runtime
from percsorenn import PercScoreProxyNet    

import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class OnnxEngine:
    def __init__(self, batch_size, model_path, onnx_file=None):
        '''
        Create an ONNX inference engine from a PyTorch model.
        '''    
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
        if onnx_file is not None:
            if onnx_file.endswith(".onnx"):
                self.onnx_path = onnx_file
            else:
                raise ValueError("ONNX file must be a .onnx file.")
        else:
            self.onnx_file = model_path.replace(".pt", ".onnx")  # If no ONNX path is provided, use the same path as the .pt model file with an .onnx extension.

        self.batch_size = batch_size
        
        
        def load_pt_model(model_path) -> PercScoreProxyNet: 
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
        
        def populate_onnx_file(pt_model, onnx_file, verbose=False):
            '''
            Convert a PyTorch model to ONNX format.

            Args:
                pt_model (PercScoreProxyNet): PyTorch model.
                onnx_path (str): Path where the ONNX model will be saved.
                verbose (bool): Whether to print onnx export process details.
            '''
            dummy_input = torch.randn(1, 7 + pt_model.label_size) 
            torch.onnx.export(pt_model, dummy_input, onnx_file, verbose=verbose)
        
        self.pt_model = load_pt_model(model_path)    
        populate_onnx_file(self.pt_model, self.onnx_file)

def build_TRT_engine(onnx_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))

    with open(onnx_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    engine_bytes = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)


def inference(onnx_file, input_data, batch_size):
    
    engine = build_TRT_engine(onnx_file)
    context = engine.create_execution_context()
    

    input_shape = (batch_size, 7 + 3)
    output_shape = (batch_size, 1)
    input_buf = trt.cuda.alloc_buffer(self.builder.max_batch_size * trt.volume(input_shape) * trt.float32.itemsize)
    output_buf = trt.cuda.alloc_buffer(self.builder.max_batch_size * trt.volume(output_shape) * trt.float32.itemsize)



    np.copyto(inputs[0].host, input_data)
    trt_outputs = common.do_inference(
        context,
        engine=trt_engine,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    return trt_outputs







def do_inference(self, inputs, outputs, bindings, stream):
    '''
    Run the engine. The output will be a 1D tensor of length 1000, where each value represents the probability that the image corresponds to that label.
    '''
    return common.do_inference(
        self.context,
        engine=self.engine,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )

def main(self):
    '''
    Run the engine on a test image.
    '''
    inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
    test_image

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(0)
config = builder.create_builder_config()
parser = trt.OnnxParser(network, TRT_LOGGER)

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
# Load the Onnx model and parse it in order to populate the TensorRT network.
with open(model_file, "rb") as model:
    if not parser.parse(model.read()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

engine_bytes = builder.build_serialized_network(network, config)
runtime = trt.Runtime(TRT_LOGGER)
return runtime.deserialize_cuda_engine(engine_bytes)


def load_normalized_test_case(test_image, pagelocked_buffer):
# Converts the input image to a CHW Numpy array
def normalize_image(image):
    # Resize, antialias and transpose the image to CHW.
    c, h, w = ModelData.INPUT_SHAPE
    image_arr = (
        np.asarray(image.resize((w, h), Image.LANCZOS))
        .transpose([2, 0, 1])
        .astype(trt.nptype(ModelData.DTYPE))
        .ravel()
    )
    # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
    return (image_arr / 255.0 - 0.45) / 0.225

# Normalize the image and copy to pagelocked memory.
np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
return test_image


def main():
# Set the data path to the directory that contains the trained models and test images for inference.
_, data_files = common.find_sample_data(
    description="Runs a ResNet50 network with a TensorRT inference engine.",
    subfolder="resnet50",
    find_files=[
        "binoculars.jpeg",
        "reflex_camera.jpeg",
        "tabby_tiger_cat.jpg",
        ModelData.MODEL_PATH,
        "class_labels.txt",
    ],
)
# Get test images, models and labels.
test_images = data_files[0:3]
onnx_model_file, labels_file = data_files[3:]
labels = open(labels_file, "r").read().split("\n")

# Build a TensorRT engine.
engine = build_engine_onnx(onnx_model_file)
# Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
# Allocate buffers and create a CUDA stream.
inputs, outputs, bindings, stream = common.allocate_buffers(engine)
# Contexts are used to perform inference.
context = engine.create_execution_context()

# Load a normalized test case into the host input page-locked buffer.
test_image = random.choice(test_images)
test_case = load_normalized_test_case(test_image, inputs[0].host)
# Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
# probability that the image corresponds to that label
trt_outputs = common.do_inference(
    context,
    engine=engine,
    bindings=bindings,
    inputs=inputs,
    outputs=outputs,
    stream=stream,
)
# We use the highest probability as our prediction. Its index corresponds to the predicted label.
pred = labels[np.argmax(trt_outputs[0])]
common.free_buffers(inputs, outputs, stream)
if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
    print("Correctly recognized " + test_case + " as " + pred)
else:
    print("Incorrectly recognized " + test_case + " as " + pred)

