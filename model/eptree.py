import random
from typing import Any, Callable, Dict, List, Optional, cast

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class TreeNode:
    def __init__(
        self,
        tree_idx: int,
        node_idx: int,
        decode_fn: Callable,
        token_id_list: List[int],
        log_prob_list: List[float],
        finish_reason: Optional[str] = None,
        is_end: bool = False,
        parent_node: Optional["TreeNode"] = None,
        parent_node_idx: Optional[int] = None,
        parent_node_split_idx: Optional[int] = None,
        child_nodes: Optional[List["TreeNode"]] = None,
        child_split_indices: Optional[List[int]] = None,
        max_length: int = 7144,
        eos_id: Optional[int] = None,
    ):
        """
        树节点的信息
        """
        # --- 分组信息 ---
        self.tree_idx: int = tree_idx
        self.node_idx: int = node_idx

        # --- 临时存储原始数据 ---
        self.token_id_list: List[int] = token_id_list
        self.log_prob_list: List[float] = log_prob_list
        self.finish_reason: Optional[str] = finish_reason
        self.is_end: bool = is_end

        # --- 父亲节点信息 ---
        self.parent_node = parent_node
        self.parent_node_idx = parent_node_idx
        self.parent_node_split_idx = parent_node_split_idx

        # --- 孩子节点信息 ---
        self.child_nodes: List["TreeNode"] = child_nodes if child_nodes else []
        self.child_split_indices: List[int] = child_split_indices if child_split_indices else []

        # --- 孩子正确率信息（分段） ---
        self.child_correct_num: List[int] = []
        self.child_total_num: List[int] = []

        # --- 步骤 1: 计算父节点（前缀）信息 ---
        self.aggregate_str: str = ""
        self.aggregate_token_ids: List[int] = []
        if parent_node is not None:
            parent_token_str_list = parent_node.token_str_list
            self.aggregate_str = parent_node.aggregate_str + "".join(
                parent_token_str_list[:parent_node_split_idx]
            )
            self.aggregate_token_ids = (
                parent_node.aggregate_token_ids + parent_node.token_id_list[:parent_node_split_idx]
            )

        # --- 步骤 2: 【核心截断逻辑】检查总长度并【物理截断或替换】 ---
        total_length = len(self.aggregate_token_ids) + len(self.token_id_list)

        # 修复“差一错误”：
        # 1. 序列过长 (total_length > max_length)
        # 2. 序列长度“刚刚好”，但是因为 "length" 停止的（意味着它没有 EOS）
        if total_length > max_length or (total_length == max_length and self.finish_reason == "length"):
            allowed_new_tokens = max(0, max_length - len(self.aggregate_token_ids))

            if allowed_new_tokens > 0 and eos_id is not None:
                # 我们的总长度必须是 max_length
                # 保留 (allowed - 1) 个 token
                self.token_id_list = self.token_id_list[: allowed_new_tokens - 1]
                self.log_prob_list = self.log_prob_list[: allowed_new_tokens - 1]

                # 手动将最后一个 token 替换为 EOS
                self.token_id_list.append(eos_id)
                if self.log_prob_list:
                    self.log_prob_list.append(self.log_prob_list[-1])
                else:
                    self.log_prob_list.append(0.0)
            else:
                # 如果 allowed_new_tokens 为 0，或者没有 eos_id，就执行简单截断
                self.token_id_list = self.token_id_list[:allowed_new_tokens]
                self.log_prob_list = self.log_prob_list[:allowed_new_tokens]

            self.is_end = True
            self.finish_reason = "length"

        # --- 步骤 3: 基于（被替换/截断的）列表生成字符串 ---
        self.token_str_list: List[str] = [decode_fn([token_id]) for token_id in self.token_id_list]
        self.token_num: int = len(self.token_id_list)

        self.total_str: str = self.aggregate_str + "".join(self.token_str_list)

        # --- 步骤 4: 掩码信息 (基于被截断的列表) ---
        self.mask: List[bool] = [False] * len(self.token_str_list)
        if len(self.aggregate_token_ids) > 0 and len(self.token_str_list) > 0:
            self.mask[0] = True

        # --- 步骤 5: 检查特殊token (基于被截断的列表) ---
        for i, token_str in enumerate(self.token_str_list):
            if "conclusion" in token_str.lower() or "answer" in token_str.lower():
                for j in range(i + 1, len(self.mask)):
                    self.mask[j] = True
                self.is_end = True
                break

        # --- 节点的分数 ---
        self.binary_score: Optional[float] = None
        self.score: Optional[float] = None

    def get_prefix(self, current_token_index: int) -> str:
        parent_tokens = self.aggregate_str
        return parent_tokens + "".join(self.token_str_list[:current_token_index])

    def get_prefix_ids(self, current_token_index: int) -> List[int]:
        parent_token_ids = self.aggregate_token_ids
        return parent_token_ids + self.token_id_list[:current_token_index]

    def add_child(self, child_node: "TreeNode", split_index: int) -> None:
        self.child_nodes.append(child_node)
        self.child_split_indices.append(split_index)
        child_node.parent_node = self
        child_node.parent_split_index = split_index  # type: ignore

    def get_max_entropy_tokens(self, top_n: int = 1) -> List[int]:
        entropies = []
        for i, log_prob in enumerate(self.log_prob_list):
            if not self.mask[i]:
                entropy = -log_prob
                entropies.append((entropy, i))
        sorted_indices = sorted(entropies, key=lambda x: x[0], reverse=True)
        result = [idx for _, idx in sorted_indices[:top_n]]
        while len(result) < top_n:
            result += result[: top_n - len(result)]
        return result


def query_local_vllm_ids_with_logprobs(
    input_ids: List[List[int]],
    llm,
    n=1,
    skip_special_tokens=True,
    max_tokens=4096,
    stops=None,
    temperature=0.9,
    top_p=0.9,
    min_tokens=0,
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        skip_special_tokens=skip_special_tokens,
        stop_token_ids=stops,
        n=n,
        logprobs=True,
        detokenize=False,
    )

    content_token_id_lists: List[List[int]] = []
    finish_reason_lists: List[str] = []
    token_num_lists: List[int] = []
    log_probs_lists: List[List[float]] = []

    vllm_inputs = [{"prompt_token_ids": input_id} for input_id in input_ids]
    outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

    for output in outputs:
        assert len(output.outputs) == 1
        out = output.outputs[0]
        log_probs_dict_lists = list(out.logprobs)

        content_token_ids = [next(iter(log_probs_dict.keys())) for log_probs_dict in log_probs_dict_lists]

        log_probs = [next(iter(log_probs_dict.values())).logprob for log_probs_dict in log_probs_dict_lists]

        finish_reasons = out.finish_reason
        token_nums = len(out.token_ids)

        content_token_id_lists.append(content_token_ids)
        finish_reason_lists.append(finish_reasons)
        token_num_lists.append(token_nums)
        log_probs_lists.append(log_probs)

    return (content_token_id_lists, finish_reason_lists, token_num_lists, log_probs_lists)


class EntropyGuidedChainLocalManager:
    def __init__(
        self,
        args: Dict[str, Any],
        llm: Any,
        encode_fn: Callable,
        decode_fn: Callable,
        eos_tokens_set: List[int],
        eos_id: Optional[int] = None,
    ):
        """
        初始化 (已移除 evaluator_urls 和 extractor_urls)
        """
        self.args = args
        self.llm = llm
        self.eos_tokens_set = eos_tokens_set
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.eos_id = eos_id

    def entropy_guided_chain(self, vllm_inputs: List[List[int]]) -> List[List[int]]:
        """
        熵引导链式推理 (已修正)
        """

        M = self.args["m"]
        N = self.args["n"]
        L = self.args["l"]
        T = self.args["t"]
        assert M * (N * L * T + 1) >= self.args["n_responses"]
        batch_size = len(vllm_inputs)
        max_length = self.args["generate_max_len"]
        tree_lists = []

        # 1. 初始采样 (M)
        initial_prompt_ids: list[list[int]] = sum([[item] * M for item in vllm_inputs], [])

        initial_results = query_local_vllm_ids_with_logprobs(
            initial_prompt_ids,
            llm=self.llm,
            skip_special_tokens=False,
            max_tokens=max_length,
            stops=self.eos_tokens_set,
            temperature=self.args["temperature"],
            top_p=self.args["top_p"],
        )

        for idx, (content_token_ids, finish_reason, _, log_probs) in enumerate(zip(*initial_results)):
            root_node = TreeNode(
                tree_idx=idx,
                node_idx=0,
                decode_fn=self.decode_fn,
                token_id_list=content_token_ids,
                log_prob_list=log_probs,
                is_end=True,
                finish_reason=finish_reason,
                max_length=max_length,
                eos_id=self.eos_id,
            )
            tree_lists.append([root_node])

        # 2. 迭代扩展 (L, N, T)
        for iteration in range(L):
            expansion_tasks = []
            for tree_idx, tree_list in enumerate(tree_lists):
                tree_entropy_tokens = []
                for node_idx, node in enumerate(tree_list):
                    if not all(node.mask):
                        if self.args["use_diverse_sampling"]:
                            entropy_tokens = node.get_max_entropy_tokens(
                                top_n=N * self.args["diverse_upsampling"]
                            )
                        else:
                            entropy_tokens = node.get_max_entropy_tokens(top_n=N)
                        for token_idx in entropy_tokens:
                            entropy_value = -node.log_prob_list[token_idx]
                            tree_entropy_tokens.append((entropy_value, tree_idx, node_idx, node, token_idx))

                tree_entropy_tokens.sort(reverse=True)

                if self.args["use_diverse_sampling"]:
                    token_indices = [token_idx for _, _, _, _, token_idx in tree_entropy_tokens]
                    scores = [entropy_value for entropy_value, _, _, _, _ in tree_entropy_tokens]
                    selected_indices = self.select_diverse_tokens(token_indices, scores, N)
                    selected_tokens = []
                    for token_idx in selected_indices:
                        for item in tree_entropy_tokens:
                            if item[4] == token_idx:
                                selected_tokens.append(item)
                                break
                    expansion_tasks.extend(
                        [
                            (tree_idx, node_idx, node, token_idx)
                            for _, tree_idx, node_idx, node, token_idx in selected_tokens
                        ]
                    )
                else:
                    expansion_tasks.extend(
                        [
                            (tree_idx, node_idx, node, token_idx)
                            for _, tree_idx, node_idx, node, token_idx in tree_entropy_tokens[:N]
                        ]
                    )

            if not expansion_tasks:
                break

            m_tree_top_n_prompt_ids = []
            task_mapping = {}
            for i, (tree_idx, node_idx, node, split_idx) in enumerate(expansion_tasks * T):
                prefix_ids = node.get_prefix_ids(split_idx)
                prompt_ids = initial_prompt_ids[tree_idx] + prefix_ids
                m_tree_top_n_prompt_ids.append(prompt_ids)
                task_mapping[i] = (tree_idx, node_idx, node, split_idx)

            inference_results = query_local_vllm_ids_with_logprobs(
                m_tree_top_n_prompt_ids,
                llm=self.llm,
                skip_special_tokens=False,
                max_tokens=max_length,
                stops=self.eos_tokens_set,
                temperature=self.args["temperature"],
                top_p=self.args["top_p"],
            )

            for i, (content_token_ids, finish_reason, _, log_probs) in enumerate(zip(*inference_results)):
                tree_idx, node_idx, parent_node, split_idx = task_mapping[i]
                new_node = TreeNode(
                    tree_idx=tree_idx,
                    node_idx=len(tree_lists[tree_idx]),
                    token_id_list=content_token_ids,
                    decode_fn=self.decode_fn,
                    log_prob_list=log_probs,
                    is_end=True,
                    parent_node=parent_node,
                    parent_node_idx=node_idx,
                    parent_node_split_idx=split_idx,
                    finish_reason=finish_reason,
                    max_length=max_length,
                    eos_id=self.eos_id,
                )
                parent_node.add_child(new_node, split_idx)
                tree_lists[tree_idx].append(new_node)

        # -----------------------------------------------------
        # 3. 随机选取 (替换了评估和 gather_paths)
        # -----------------------------------------------------

        all_nodes: list[TreeNode] = sum(tree_lists, [])
        assert all(node.is_end for node in all_nodes)

        nodes_sample_grouped: list[list[TreeNode]] = [[] for _ in range(batch_size)]
        for node in all_nodes:
            nodes_sample_grouped[node.tree_idx // M].append(node)

        selected_responses_ids = []
        for nodes in nodes_sample_grouped:
            assert len(nodes) == M * (N * L * T + 1)
            selected_nodes = random.sample(nodes, self.args["n_responses"])
            # --- 关键修复：打包 Token ID 列表, 而不是字符串 ---
            for node in selected_nodes:
                # full_response_ids 是被 TreeNode 截断/替换过的
                # 它代表了 *仅回复* 部分 (aggregate_token_ids + token_id_list)
                full_response_ids = node.aggregate_token_ids + node.token_id_list
                selected_responses_ids.append(full_response_ids)

        return selected_responses_ids

    def select_diverse_tokens(self, token_indices, scores, n):
        """
        select the most diverse n tokens from the top-k tokens
        """
        if n == 0 or len(token_indices) <= n:
            return token_indices

        import numpy as np

        tokens = np.array(token_indices)
        selected = [0]
        remaining = list(range(1, len(tokens)))

        while len(selected) < n:
            max_min_dist = -float("inf")
            best_idx = -1
            for i in remaining:
                min_dist = float("inf")
                for j in selected:
                    dist = abs(int(tokens[i]) - int(tokens[j]))
                    min_dist = min(min_dist, dist)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            if best_idx == -1:  # 以防万一所有 token 都相同
                best_idx = remaining[0]

            selected.append(best_idx)
            remaining.remove(best_idx)
        return [token_indices[i] for i in selected]


# ---------------------------------------------------------------------------
# 步骤3：EPTree 类 (已修正)
# ---------------------------------------------------------------------------


class EPTree:
    def __init__(
        self,
        n_responses: int,
        tokenizer,
        vllm_engine: LLM,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 100,
        m: int = 4,
        n: int = 4,
        l: int = 3,
        t: int = 2,
        **kwargs,
    ):
        self.n_responses = n_responses
        self.tokenizer = tokenizer

        # --- 关键修复：恢复 manager_args 的完整定义 ---
        self.manager_args = {
            "m": m,
            "n": n,
            "l": l,
            "t": t,
            "temperature": temperature,
            "top_p": top_p,
            "generate_max_len": max_new_tokens,
            "n_responses": self.n_responses,
            "use_diverse_sampling": kwargs.get("use_diverse_sampling", True),
            "diverse_upsampling": kwargs.get("diverse_upsampling", 5),
        }
        self.manager_args.update(kwargs)

        def simple_encode_fn(prompt, **kwargs):
            return self.tokenizer(prompt, **kwargs)

        def simple_decode_fn(token_ids, **kwargs):
            return self.tokenizer.decode(token_ids, skip_special_tokens=True, **kwargs)

        self.manager = EntropyGuidedChainLocalManager(
            args=self.manager_args,
            llm=vllm_engine,
            encode_fn=simple_encode_fn,
            decode_fn=simple_decode_fn,
            eos_tokens_set=[tokenizer.eos_token_id],
            eos_id=tokenizer.eos_token_id,  # <--- 关键修复: 传递 eos_id
        )

    def generate_sequences(self, vllm_inputs: List[List[int]]) -> list[list[int]]:
        return self.manager.entropy_guided_chain(vllm_inputs)


# ---------------------------------------------------------------------------
# 步骤4：你的新 main 函数 (已修正)
# ---------------------------------------------------------------------------


def main():
    model_id = "/inspire/hdd/global_user/weizhongyu-24036/effciency_workspace/models/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = "<|image_pad|>"
    tokenizer.pad_token_id = 151655
    print(f"Set pad_token to '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id}) for batching.")

    prompts = ["Why is the sky blue?", "What is the capital of France?"]
    input_ids = tokenizer(prompts)["input_ids"]

    llm = LLM(model_id, trust_remote_code=True, enforce_eager=True)

    eptree = EPTree(
        n_responses=10,
        tokenizer=tokenizer,
        vllm_engine=llm,
        max_new_tokens=100,  # <--- "回复" 的最大长度
        m=2,  # 初始树 (M)
        n=4,  # 扩展节点 (N)
        l=1,  # 迭代 (L)
        t=1,  # 扩展次数 (T)
    )

    # generate_sequences 现在 *只* 返回回复
    responses = eptree.generate_sequences(input_ids)

    print(f"\n--- Total sequences generated: {len(responses)} ---")

    assert len(responses) == 2 * eptree.n_responses

    # Check: if a sequence contains padding, the last non-pad token must be EOS
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    for row_idx, response in enumerate(responses):
        seq = torch.tensor(response)
        if not (seq == pad_id).any():
            continue
        non_pad_positions = (seq != pad_id).nonzero(as_tuple=False).flatten()
        assert non_pad_positions.numel(), f"sequence {row_idx} contains no non-padding tokens"
        last_non_pad_idx = cast(int, non_pad_positions[-1].item())
        assert int(seq[last_non_pad_idx].item()) == int(
            eos_id
        ), f"sequence {row_idx}: last non-pad token {int(seq[last_non_pad_idx].item())} != EOS {int(eos_id)}"

    print(f"期望的序列数: {len(prompts) * eptree.n_responses} (如果树搜索结果充足)")

    print("\n--- Decoding Final Selected Sequences ---")
    response_strs = tokenizer.batch_decode(responses)
    for i, response_str in enumerate(response_strs):
        print(f"\n[Response {i+1}]:")
        print(response_str)
        print("-" * 20)


if __name__ == "__main__":
    main()
