class PromptFormatter:
    system_prompt_templates = {
        "general": "You are a helpful assistant.",
        "reasoning": "You are a helpful assistant. "
        "You first think about the question step by step and then provide the user with the answer.",
        "none": "",
    }

    cot_instruction_templates = {
        "plain": " Show your work in {start_marker} {end_marker} tags, "
        "and return the final answer in \\boxed{{...}}.",
        "instruct": " First think about the question step by step in one "
        "`{start_marker}...{end_marker}` block, "
        "and then provide the final answer in the format of `\\boxed{{...}}`. "
        "Example output format (note that you should put real thinking and answer instead of placeholders): "
        "`{start_marker}...{end_marker}\\boxed{{...}}`.",
        "xml": " Show your work in one `<think>...</think>` block, "
        "and return the final answer in `<answer>...</answer>`. ",
        "latent": "",  # since latent model needs to be trained, we skip the format instruction
        "none": "",
    }

    dataset_prompt_templates = {
        "countdown": "Using the numbers {numbers}, create an expression that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once."
        "{format_instruction} An effective final answer should be an expression like (1 + 2) / 3.",
        "dapomath": "{question}{format_instruction} You should only provide digits as the final answer.",
        "aime": "{question}{format_instruction} You should only provide digits as the final answer.",
        "math500": "{question}{format_instruction} You should provide digits or a latex expression as the final answer.",
    }

    def __init__(self, dataset_name: str, cot_type: str, plain_start_marker: str, plain_end_marker: str):
        self.cot_type = cot_type
        self.plain_start_marker = plain_start_marker
        self.plain_end_marker = plain_end_marker
        self.format_instruction = self.cot_instruction_templates[cot_type].format(
            start_marker=plain_start_marker, end_marker=plain_end_marker
        )
        self.template = self.dataset_prompt_templates[dataset_name]

    def __call__(self, **kwargs) -> str:
        return self.template.format(format_instruction=self.format_instruction, **kwargs)

    @classmethod
    def get_system_prompt(cls, system_prompt_type: str) -> str:
        return cls.system_prompt_templates[system_prompt_type]
