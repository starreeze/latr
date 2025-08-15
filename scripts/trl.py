import datetime
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import cast

from transformers.hf_argparser import HfArgumentParser


@dataclass
class TRLRunnerArgs:
    gpu_ids: list[str] = field(
        default_factory=list,
        metadata={"help": "List of GPU IDs to use. Should be seperated by space or comma."},
    )
    vllm_n_gpu: int = field(
        default=0, metadata={"help": "Number of GPUs to run the vLLM serve on. If 0, do not use vllm."}
    )
    port: int = field(default=29501, metadata={"help": "Port to run the accelerate on."})
    model: str = field(default="Qwen/Qwen2.5-3B", metadata={"help": "Model to use."})
    max_completion_length: int = field(default=1024, metadata={"help": "Max completion length."})
    deepspeed: str = field(default="zero2", metadata={"help": "Deepspeed zero stage."})


class TRLRunner:
    """Class to manage the TRL training and vLLM serving process."""

    def __init__(self):
        parser = HfArgumentParser([TRLRunnerArgs])  # type: ignore

        # Parse known args first
        self.args, self.remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        self.args = cast(TRLRunnerArgs, self.args)

        if len(self.args.gpu_ids) == 1:
            split_char = "," if "," in self.args.gpu_ids[0] else " "
            self.args.gpu_ids = self.args.gpu_ids[0].strip(" ,").split(split_char)

    def start_vllm_server(self, gpu_ids: list[str]) -> None:
        """Start the vLLM server on the specified GPU."""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Create a timestamped log file for vLLM
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        vllm_log_file = f"logs/vllm_server_{timestamp}.log"

        # Start vllm-serve in the background on the specified GPU
        vllm_serve_cmd = [
            "trl",
            "vllm-serve",
            "--model",
            self.args.model,
            "--max_model_len",
            str(self.args.max_completion_length),
            "--tensor_parallel_size",
            str(len(gpu_ids)),
        ]

        print(f"Running command: CUDA_VISIBLE_DEVICES={','.join(gpu_ids)} {' '.join(vllm_serve_cmd)}")
        print(f"vLLM server output will be written to: {vllm_log_file}")

        vllm_env = os.environ.copy()
        vllm_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        vllm_env["NCCL_DEBUG"] = "WARN"

        with open(vllm_log_file, "w") as log_file:
            self.vllm_process = subprocess.Popen(
                vllm_serve_cmd,
                env=vllm_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            )

    def build_accelerate_command(self, gpu_ids: list[str]) -> list[str]:
        """Build the accelerate command with all necessary arguments."""
        return [
            "accelerate",
            "launch",
            "--config_file",
            f"configs/{self.args.deepspeed}.yaml",
            "--num_processes",
            str(len(gpu_ids)),
            "--main_process_port",
            str(self.args.port),
            "-m",
            "train.trl.grpo",
            "--model",
            self.args.model,
            "--max_completion_length",
            str(self.args.max_completion_length),
            "--use_vllm",
            "true" if self.args.vllm_n_gpu else "false",
            *self.remaining_args,
        ]

    def run_accelerate_training(self, gpu_ids: list[str]) -> int:
        """Run the accelerate training process."""
        accelerate_cmd = self.build_accelerate_command(gpu_ids)
        print(f"Running command: CUDA_VISIBLE_DEVICES={','.join(gpu_ids)} {' '.join(accelerate_cmd)}")

        try:
            # for grpo, this is necessary to avoid OOM
            env = os.environ.copy()
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["DEEPSPEED_TIMEOUT"] = "5400"
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            self.accelerate_process = subprocess.Popen(accelerate_cmd, env=env)
            self.accelerate_process.wait()

            if self.accelerate_process.returncode == 0:
                print("Accelerate training completed successfully.")
                return 0
            else:
                print(
                    f"Accelerate training failed with error code: {self.accelerate_process.returncode}",
                    file=sys.stderr,
                )
                return 1
        except Exception as e:
            print(f"Accelerate training failed with error: {e}", file=sys.stderr)
            return 1

    @staticmethod
    def cleanup(name: str, process) -> None:
        """Clean up processes and resources."""
        if process:
            print(f"Terminating {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"{name} did not terminate gracefully, killing.")
                process.kill()
            print(f"{name} stopped.")

    def run(self) -> int:
        """Main method to run the entire process."""
        os.environ["NCCL_CUMEM_ENABLE"] = "0"

        if self.args.vllm_n_gpu == 0:
            accelerate_gpu_ids = self.args.gpu_ids
        else:
            # First GPU for vLLM, rest for accelerate
            accelerate_gpu_ids = self.args.gpu_ids[self.args.vllm_n_gpu :]
            vllm_gpu_id = self.args.gpu_ids[: self.args.vllm_n_gpu]
            self.start_vllm_server(vllm_gpu_id)
        try:
            return self.run_accelerate_training(accelerate_gpu_ids)
        finally:
            try:
                self.cleanup("vllm server", self.vllm_process)
            except AttributeError:
                pass
            self.cleanup("accelerate trainer", self.accelerate_process)


def main() -> int:
    """Entry point for the script."""
    runner = TRLRunner()
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
