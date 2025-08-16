from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from model.generation import KeyTokenGenMixin


class Gpt2KeyTokenModel(KeyTokenGenMixin, GPT2LMHeadModel):
    pass


class LlamaKeyTokenModel(KeyTokenGenMixin, LlamaForCausalLM):
    pass


class Qwen2KeyTokenModel(KeyTokenGenMixin, Qwen2ForCausalLM):
    pass


class Qwen3KeyTokenModel(KeyTokenGenMixin, Qwen3ForCausalLM):
    pass
