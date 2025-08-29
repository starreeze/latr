import sys

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

base_model = "/inspire/hdd/global_user/weizhongyu-24036/effciency_workspace/models/Qwen2.5-3B"

if __name__ == "__main__":
    local_dir = sys.argv[1]
    target_dir = sys.argv[2]
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=local_dir,
        target_dir=target_dir,
        hf_model_config_path=base_model,
    )
    merger = FSDPModelMerger(config)
    merger.merge_and_save()
