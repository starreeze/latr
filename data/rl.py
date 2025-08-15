from data.base import DataBase

# def extract_hash_answer(text: str) -> str | None:
#     if "####" not in text:
#         return None
#     return text.split("####")[1].strip().replace(",", "")


# def load_gsm8k(
#     data_path: str,
#     split: str,
#     n_train_samples: int,
#     n_eval_samples: int,
#     cot_type: str,
#     system_prompt_type: str,
#     *args,
#     **kwargs,
# ) -> Dataset:
#     data = load_raw_gsm8k(data_path, split)
#     if split == "test":
#         assert len(data) >= n_eval_samples
#         if n_eval_samples >= 0:
#             data = data.select(range(n_eval_samples))
#     elif split == "train":
#         if n_train_samples >= 0:
#             data = data.select(range(min(len(data), n_train_samples)))
#     else:
#         raise ValueError(f"Invalid split: {split}")

#     system_prompt_message = prompts.get_system_prompt_message(system_prompt_type)
#     data = data.map(
#         lambda x: {
#             "prompt": [
#                 *system_prompt_message,
#                 {"role": "user", "content": prompts.get_digit_math_user_prompt(x["question"], cot_type)},
#             ],
#             "answer": extract_hash_answer(x["answer"]),
#         }
#     )
#     return data


class RLData(DataBase):
    def process_countdown_sample(self, x: dict) -> dict:
        plain_prompt = self.prompt_formatter(numbers=x["nums"], target=x["target"])
        message_prompt = [*self.system_prompt_message, {"role": "user", "content": plain_prompt}]
        return {
            "prompt": message_prompt if self.template == "conv" else plain_prompt,
            "answer": {"target": x["target"], "numbers": x["nums"]},
        }

    def process_gsm8k_sample(self, x: dict) -> dict:
        # TODO split train and test
        raise NotImplementedError("gsm8k is not supported")

    def process_aime_sample(self, x: dict) -> dict:
        plain_prompt = self.prompt_formatter(question=x["question"])
        message_prompt = [*self.system_prompt_message, {"role": "user", "content": plain_prompt}]
        return {"prompt": message_prompt if self.template == "conv" else plain_prompt, "answer": x["answer"]}

    def process_dapomath_sample(self, x: dict) -> dict:
        plain_prompt = self.prompt_formatter(question=x["prompt"])
        message_prompt = [*self.system_prompt_message, {"role": "user", "content": plain_prompt}]
        return {
            "prompt": message_prompt if self.template == "conv" else plain_prompt,
            "answer": x["solution"],
        }

    def process_math500_sample(self, x: dict) -> dict:
        plain_prompt = self.prompt_formatter(question=x["problem"])
        message_prompt = [*self.system_prompt_message, {"role": "user", "content": plain_prompt}]
        return {"prompt": message_prompt if self.template == "conv" else plain_prompt, "answer": x["answer"]}
