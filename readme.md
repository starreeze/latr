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

```bash
python -m tools.diversity --model Qwen/Qwen2.5-3B --force_key_token_model qwen2 --eval_dataset countdown --do_sample --temperature 0.6 --top_k 20 --top_p 0.95 --cot_type xml --max_new_tokens 1000 --mix_ratio 0.5 --n_gpus 8 --generate_batch_size 2 --num_return_sequences 4
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

## Run RL

### Data

```bash
python -m data.countdown_gen --src SRC --dst DST --template_type TEMPLATE_TYPE
```

- src: the source dataset name or a local path, default to `Jiayi-Pan/Countdown-Tasks-3to4`
- dst: the destination path to save the verl-ready dataset, default to `dataset/countdown`
- template_type: the template type (`base`, `qwen-instruct`), default to `base`

### Train

scripts are provided in `scripts`.

## Plot results

```bash
python tools/plot.py --path dataset/countdown.csv --interval 50 --min_y 0.4 --max_y 0.8 --y_tick 0.4 0.5 0.6 0.7 0.8
python tools/plot.py --path dataset/math.csv --interval 40
python tools/plot.py --path dataset/k.csv --min_y 58 --set_label 0
python tools/plot.py --path dataset/temp.csv --set_label 0 --legend lower_left --y_tick 40 50 60 70 80
python tools/plot_single.py --path dataset/math_dapo.csv --interval 40
```

## Contributing

Fork and open a pull request. Follow the instructions below or your PR will fail.

1. **Static linting.** Use _Pylance_ (basic level) to lint your code while doing your work. Please also fix all warnings. Refer to https://docs.pydantic.dev/latest/integrations/visual_studio_code/#configure-vs-code to configure your VSCode. NOTE: Be cautious of using `# type: ignore` to suppress type errors, as you may be ignoring valuable traces of bugs; usually typing.cast() is more preferred. If you want to add external modules which will not pass the linter, you can add them to `pyrightconfig.json`.
2. **Code formatting.** Config your vscode to use _black_ to do code formatting. The arguments are supposed to be:
   ![](docs/assets/black.png)
   If you do not like this code style or you cannot complete the config, you can also use command line to format your code before opening a PR:
   ```shell
   pip install black==24.10.0
   black . --skip-magic-trailing-comma --line-length 110
   ```
3. **Import sorting.** Install _isort_ extension in your vscode and run `isort` to sort your imports automatically, or run this before opening a PR:
   ```shell
   pip install isort==6.0.1
   isort . --profile black  --line-length 110
   ```
4. **Code style checking.** Config your vscode to use _flake8_ to check your code style. The arguments are supposed to be:
   ![](docs/assets/flake8.png)
   If you do not like this code style or you cannot complete the config, you can also use command line to check your code style before opening a PR:
   ```shell
   pip install flake8==7.2.0
   flake8 . --ignore=E501,E203,E266,E731,E704,W503
   ```
