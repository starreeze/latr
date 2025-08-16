# Key-Token Tree Search Sampling & RL

Updates on progress and full experiment logs: [Feishu Doc](https://ewtmzxnm3yv.feishu.cn/docx/B9xcdAHanodGt3xaeDoc6sx5nIe)


## Installation

```bash
pip install -r requirements.txt
```

## sft
just run
```
bash scripts/sft.sh
```
default model is qwen3-1.7b-base and dataset is countdown.


## sampling strategy

### evaluation

#### eval explicit cot

```bash
python -m tools.eval --model /home/xingsy/data_91/model/Qwen3-1.7B --eval_dataset aime --do_sample --temperature 0.6 --top_k 20 --top_p 0.95 --cot_type plain --max_think_tokens 7500 --max_new_tokens 8192 --enable_latent false [--system_prompt_type none] --n_gpus 8 --generate_batch_size 16
```

## the divergence model

### data

```bash
python -m data.diverge --model /home/nfs04/model/Qwen2.5/Qwen2.5-1.5B-Instruct --dataset_name countdown --do_sample --temperature 0.6 --top_k 20 --top_p 0.95 --cot_type plain --max_new_tokens 1000 --enable_latent false --force_key_token_model qwen2 --num_return_sequences 32 --n_train_samples 9024 --start_pos 8000 --n_gpus 8
```

### train

```bash
bash scripts/diverge.sh
```

### eval

# Contributing

Fork and open a pull request. Follow the instructions below or your PR will fail.

1. **Static linting.** Use *Pylance* (basic level) to lint your code while doing your work. Please also fix all warnings. Refer to https://docs.pydantic.dev/latest/integrations/visual_studio_code/#configure-vs-code to configure your VSCode. NOTE: Be cautious of using `# type: ignore` to suppress type errors, as you may be ignoring valuable traces of bugs; usually typing.cast() is more preferred. If you want to add external modules which will not pass the linter, you can add them to `pyrightconfig.json`.
2. **Code formatting.** Config your vscode to use *black* to do code formatting. The arguments are supposed to be:
   ![](docs/assets/black.png)
   If you do not like this code style or you cannot complete the config, you can also use command line to format your code before opening a PR:
   ```shell
   pip install black==24.10.0
   black . --skip-magic-trailing-comma --line-length 110
   ```
3. **Import sorting.** Install *isort* extension in your vscode and run `isort` to sort your imports automatically, or run this before opening a PR:
   ```shell
   pip install isort==6.0.1
   isort . --profile black  --line-length 110
   ```
4. **Code style checking.** Config your vscode to use *flake8* to check your code style. The arguments are supposed to be:
   ![](docs/assets/flake8.png)
   If you do not like this code style or you cannot complete the config, you can also use command line to check your code style before opening a PR:
   ```shell
   pip install flake8==7.2.0
   flake8 . --ignore=E501,E203,E266,E731,E704,W503
   ```