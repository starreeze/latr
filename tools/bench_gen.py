import time
from typing import cast

# import torch
import torch
from transformers.hf_argparser import HfArgumentParser

from args import DataArgs, GenerationArgs, ModelArgs, TrainingArgs
from data.loader import get_sft_collator
from data.loader import load as load_datasets
from model.loader import load as load_model
from tools.utils import set_seed

# from tools.interpret import get_token_probs, get_topk_tokens, print_list_dict


def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, GenerationArgs, TrainingArgs))  # type: ignore
    model_args, data_args, generation_args, training_args = parser.parse_args_into_dataclasses()
    model_args = cast(ModelArgs, model_args)
    data_args = cast(DataArgs, data_args)
    generation_args = cast(GenerationArgs, generation_args)
    training_args = cast(TrainingArgs, training_args)

    set_seed(training_args.seed)
    model, tokenizer = load_model(model_args)
    model.to("cuda")  # type: ignore
    model.eval()

    if generation_args.compile_generation:
        current = time.time()
        print("start compiling")
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model, mode="reduce-overhead")
        print(f"time taken to compile: {time.time() - current}")
    _, test_dataset = load_datasets(data_type="sft", **data_args.__dict__)

    data_args.align_label_with_tf_mask = False
    collator = get_sft_collator(tokenizer, data_args)
    sample = collator([test_dataset[i] for i in range(data_args.generate_batch_size)])
    question_ids = sample["question_ids"]
    current = time.time()
    print("start generation")
    output = model.generate(  # type: ignore
        question_ids.to("cuda"),
        max_new_tokens=generation_args.max_new_tokens,
        do_sample=generation_args.do_sample,
    )
    print(tokenizer.batch_decode(output, skip_special_tokens=True))
    print(f"time taken: {time.time() - current}")


if __name__ == "__main__":
    main()
