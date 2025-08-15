import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.multiprocessing as mp
from datasets import Dataset


class MultiGPUWorker(ABC):
    """Abstract base class for multi-GPU workers."""

    @abstractmethod
    def process_chunk(self, dataset_chunk: Dataset, gpu_id: int, device: str, **kwargs) -> Any:
        """Process a chunk of data on a specific GPU.

        Args:
            dataset_chunk: The chunk of dataset to process
            gpu_id: The GPU ID to use
            device: The device string (e.g., "cuda:0")
            **kwargs: Additional arguments passed from the manager

        Returns:
            Any result that can be combined with other results
        """
        pass

    @abstractmethod
    def combine_results(self, results: list[Any]) -> Any:
        """Combine results from multiple workers.

        Args:
            results: List of results from each worker

        Returns:
            Combined result
        """
        pass


class MultiGPUManager:
    """Manager for multi-GPU processing using multiprocessing."""

    def __init__(self, worker_class: type[MultiGPUWorker]):
        """Initialize the manager with a worker class.

        Args:
            worker_class: Class that implements MultiGPUWorker interface
        """
        self.worker_class = worker_class

    def run(self, dataset: Dataset, n_gpus: int, **kwargs) -> Any:
        """Run processing on multiple GPUs.

        Args:
            dataset: Dataset to process
            n_gpus: Number of GPUs to use
            **kwargs: Additional arguments to pass to workers

        Returns:
            Combined result from all workers
        """
        if n_gpus <= 0:
            raise ValueError("n_gpus must be positive")

        if n_gpus == 1:
            return self._run_single_gpu(dataset, **kwargs)
        else:
            return self._run_multi_gpu(dataset, n_gpus, **kwargs)

    def _run_single_gpu(self, dataset: Dataset, **kwargs) -> Any:
        """Run processing on a single GPU."""
        print("Using single GPU process.")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        worker = self.worker_class()
        print(f"Processing {len(dataset)} samples on {device}")
        result = worker.process_chunk(dataset, 0, device, **kwargs)
        # For consistency with multi-GPU path, wrap in list and call combine_results
        return worker.combine_results([result])

    def _run_multi_gpu(self, dataset: Dataset, n_gpus: int, **kwargs) -> Any:
        """Run processing on multiple GPUs using multiprocessing."""
        print(f"Using {n_gpus} GPU processes.")

        # Set multiprocessing start method
        try:
            mp.set_start_method("spawn", force=True)
            print("Set multiprocessing start method to 'spawn'.")
        except RuntimeError as e:
            print(f"Warning: Could not set start method: {e}. Using default.")

        # Split dataset into chunks
        num_samples = len(dataset)
        chunk_size = math.ceil(num_samples / n_gpus)
        dataset_chunks = [
            dataset.select(range(i * chunk_size, min((i + 1) * chunk_size, num_samples)))
            for i in range(n_gpus)
        ]
        print(f"Split dataset into {n_gpus} chunks of approx size {chunk_size}.")

        # Create tasks for each worker
        tasks = [(gpu_id, dataset_chunks[gpu_id], self.worker_class, kwargs) for gpu_id in range(n_gpus)]

        # Run multiprocessing
        print("Starting multiprocessing pool...")
        with mp.Pool(processes=n_gpus) as pool:
            results = list(pool.imap(_worker_wrapper, tasks))

        print("Multiprocessing finished. Combining results...")
        worker = self.worker_class()
        combined_result = worker.combine_results(results)
        print("Results combined.")

        return combined_result


def _worker_wrapper(args_tuple):
    """Wrapper function for multiprocessing worker.

    This function exists at module level to be picklable for multiprocessing.
    """
    gpu_id, dataset_chunk, worker_class, kwargs = args_tuple
    device = f"cuda:{gpu_id}"
    print(f"Worker {gpu_id}: Starting processing on {device}")

    # Create worker instance and process chunk
    worker = worker_class()
    result = worker.process_chunk(dataset_chunk, gpu_id, device, **kwargs)

    print(f"Worker {gpu_id}: Finished processing")
    return result
