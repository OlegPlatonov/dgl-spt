import multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Ensure compatibility on all platforms

import numpy as np
import torch

lock = mp.Lock()
import os

os.environ["OMP_NUM_THREADS"] = "1"

def worker(memmap_path, start_index, end_index, value, shape):
    """
    Worker function to write to a specific segment of the memmap file.

    Args:
        memmap_path (str): Path to the memory-mapped file.
        start_index (int): Starting index of the segment.
        end_index (int): Ending index of the segment.
        value (float): Value to write into the segment.
        shape (tuple): Shape of the full memmap array.
    """
    global lock
    # Open the memory-mapped file
    memmap = np.memmap(memmap_path, dtype='float32', mode='r+', shape=shape)
    # Convert to PyTorch tensor
    tensor = torch.from_numpy(memmap)
    print("Before: ", tensor)

    # Write the value to the specified segment
    with lock:
        tensor[start_index:end_index, ...] = value
        print("After: ", tensor)
        # Explicitly flush changes
        memmap.flush()

def main():
    memmap_path = "/mnt/ar_hdd/fvelikon/graph-time-series/datasets/music/spatiotemporal_features/test.memmap"
    shape = (1000, 1000, 30)  # Example shape of the array
    dtype = 'float32'

    # Create the memory-mapped file
    memmap = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=shape)
    # memmap[:] = 0  # Initialize with zeros

    # Define worker arguments
    num_processes = 4
    segment_size = shape[0] // num_processes

    with mp.Pool(processes=num_processes) as pool:
        
        arguments = []
        for i in range(num_processes):
            
            start_index = i * segment_size
            end_index = (i + 1) * segment_size if i != num_processes - 1 else shape[0]
            value = i + 1  # Assign a unique value for each segment
            arguments.append((
                memmap_path,
                start_index, 
                end_index,
                value,
                shape,
                
            ))
        _ = pool.starmap(worker, arguments)

    memmap.flush()

    # Verify the result
    final_array = np.memmap(memmap_path, dtype=dtype, mode='r', shape=shape)
    print("Final array:", final_array[:])

if __name__ == '__main__':
    main()
