from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from args import ModelArgs
from model import backbone


def load(model_args: ModelArgs) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    path = model_args.model
    key_token_name = model_args.force_key_token_model

    if key_token_name is None:
        # remove device_map parameter to fix this error:
        # "AssertionError: found no DeviceMesh from dtensor args for c10d.broadcast_.default!"
        # refer to https://github.com/huggingface/accelerate/issues/3486
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=model_args.dtype,
            local_files_only=model_args.local_files_only,
            attn_implementation=model_args.attention_implementation,
            # fix generation padding issue, see https://github.com/huggingface/trl/issues/3034
            use_cache=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=model_args.local_files_only)
        if "gpt2" in path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    key_token_cls = getattr(backbone, f"{key_token_name.capitalize()}KeyTokenModel")
    model = key_token_cls.from_pretrained(
        path,
        torch_dtype=model_args.dtype,
        local_files_only=model_args.local_files_only,
        attn_implementation=model_args.attention_implementation,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=model_args.local_files_only)
    return model, tokenizer
