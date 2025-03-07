from torch2trt import torch2trt
import tensorrt as trt
import torch
import percscorenn
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_STATE_PATH = "/home/lenman/capstone/parallelrm/models/percscore-nov12-50k.pt"
BATCH_SIZES = [1, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
NUM_RUNS = 100
WARMUP_RUNS = 5
VERBOSE = True


# TODO: model.eval(), torch.no_grad(), discrete batch sizes, one time model conversion/dynamic batching on conversion


def run_inference_benchmark(model_state_path, batch_sizes, num_runs, dataframe : pd.DataFrame, warmup_runs=5, verbose = False):
   
    for batch_size in batch_sizes:
        if batch_size % 10 == 0 and verbose:
            print(f"Running inference benchmark for batch size: {batch_size}")

        x = torch.ones((batch_size, 7+3)).cuda()  # Dummy input data for trt creation 
        model = percscorenn.PercScoreProxyNet(label_size=3).cuda()
        model.load_state_dict(torch.load(model_state_path))
        
        # Convert to TensorRT and create random input data
        model_trt = torch2trt(model, [x])
        
        x = torch.rand((batch_size, 7+3)).cuda()
        
        # Warm-up runs (to stabilize GPU performance)
        for _ in range(warmup_runs):
            _ = model(x)
            _ = model_trt(x)
        
        torch.cuda.synchronize()  # Ensure all operations are completed
        
        
            
        # Collect timing results
        pytorch_times = []
        tensorrt_times = []
        
        if verbose:
            print(f"Benchmarking with batch size: {batch_size}, runs: {num_runs}")
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            y = model(x)
            torch.cuda.synchronize()
            pytorch_times.append(time.time() - start)
            
            torch.cuda.synchronize()
            start = time.time()
            y_trt = model_trt(x)
            torch.cuda.synchronize()
            tensorrt_times.append(time.time() - start)
        
        # Add results to dataframe for the specified batch size
        dataframe = dataframe.append({
            'Batch Size': batch_size,
            'PyTorch Mean (ms)': np.mean(pytorch_times)*1000,
            'TensorRT Mean (ms)': np.mean(tensorrt_times)*1000,
            'PyTorch Median': np.median(pytorch_times)*1000,
            'TensorRT Median': np.median(tensorrt_times)*1000,
            'PyTorch StdDev': np.std(pytorch_times)*1000,
            'TensorRT StdDev': np.std(tensorrt_times)*1000,
            'PyTorch Min': np.min(pytorch_times)*1000,
            'TensorRT Min': np.min(tensorrt_times)*1000,
            'PyTorch Max': np.max(pytorch_times)*1000,
            'TensorRT Max': np.max(tensorrt_times)*1000,
            'Speedup': np.mean(pytorch_times) / np.mean(tensorrt_times)
        }, ignore_index=True)
       
    return dataframe

# Example usage
if __name__ == "__main__":
    
    # Create a dataframe to store results
    dataframe = pd.DataFrame(columns=['Batch Size',
                                      'PyTorch Mean (ms)',
                                      'TensorRT Mean (ms)', 
                                      'PyTorch Median',
                                      'TensorRT Median', 
                                      'PyTorch StdDev', 
                                      'TensorRT StdDev',
                                      'PyTorch Min', 
                                      'TensorRT Min', 
                                      'PyTorch Max', 
                                      'TensorRT Max',
                                      'Speedup'])
                                      
    # Run benchmark
    dataframe = run_inference_benchmark(MODEL_STATE_PATH, BATCH_SIZES, NUM_RUNS, dataframe, verbose=VERBOSE)
    

    # Optional: Save results to CSV
    
    dataframe.to_csv('inference_benchmark_results.csv', index=False)
    print("Results saved to inference_benchmark_results.csv")



