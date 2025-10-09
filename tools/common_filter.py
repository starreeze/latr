import itertools
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm


class DiverseFilter:
    def __init__(self, k: int):
        self.k = k

    def __call__(self, sequences: torch.Tensor, pad_id: int, max_workers=1) -> list[list[int]]:
        """
        Select k most diverse sequences (right-padded).
        This implementation uses the official rouge_score library to calculate
        pairwise dissimilarity and a greedy algorithm to select the final set.

        Args:
            sequences: (batch, n_cand, seq_len) with n_cand >= k
            pad_id: padding id. Only compare tokens that are not padded.

        Returns:
            lists of indices to keep with shape (batch, k)
        """
        if max_workers <= 1:
            return [
                self._process_single(seqs.tolist(), pad_id)
                for seqs in tqdm(sequences, desc="Discretizing sequences")
            ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_single, seqs.tolist(), pad_id) for seqs in sequences]
            results = [future.result() for future in tqdm(futures, desc="Discretizing sequences")]
        return results

    def _process_single(self, sequences: list[list[int]], pad_id: int) -> list[int]:
        # 预处理序列，为 ROUGE 和 BLEU 准备不同格式
        sequences_for_rouge = []  # ROUGE 需要空格分隔的字符串
        sequences_for_bleu = []  # BLEU 需要 token 列表
        for seq in sequences:
            # 移除 padding
            unpadded_tokens = [str(token) for token in seq if token != pad_id]
            sequences_for_rouge.append(" ".join(unpadded_tokens))
            sequences_for_bleu.append(unpadded_tokens)

        # 初始化 ROUGE 和 BLEU 的 scorer
        rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        bleu_smoother = SmoothingFunction()

        num_sequences = len(sequences)
        dissimilarity_matrix = torch.zeros((num_sequences, num_sequences))

        # 计算合并后的不相似度矩阵
        for i, j in itertools.combinations(range(num_sequences), 2):
            # --- 计算 ROUGE 相似度 ---
            rouge_sim = 0.0
            str_i_rouge = sequences_for_rouge[i]
            str_j_rouge = sequences_for_rouge[j]
            if str_i_rouge and str_j_rouge:
                try:
                    rouge_scores = rouge_scorer_obj.score(str_i_rouge, str_j_rouge)
                    rouge_sim = sum(score.fmeasure for score in rouge_scores.values()) / len(rouge_scores)
                except Exception as e:
                    warnings.warn(f"ROUGE calculation failed for pair ({i}, {j}): {e}")

            # --- 计算 BLEU 相似度 ---
            bleu_sim = 0.0
            seq_i_bleu = sequences_for_bleu[i]
            seq_j_bleu = sequences_for_bleu[j]
            if seq_i_bleu and seq_j_bleu:
                # BLEU 是非对称的，我们计算两个方向的平均值
                score1 = sentence_bleu([seq_j_bleu], seq_i_bleu, smoothing_function=bleu_smoother.method7)
                score2 = sentence_bleu([seq_i_bleu], seq_j_bleu, smoothing_function=bleu_smoother.method7)
                bleu_sim: float = (score1 + score2) / 2.0  # type: ignore

            # --- 合并分数并计算不相似度 ---
            final_similarity = (rouge_sim + bleu_sim) / 2.0
            dissimilarity = 1.0 - final_similarity
            dissimilarity_matrix[i, j] = dissimilarity
            dissimilarity_matrix[j, i] = dissimilarity

        # 选择 k 个最不相似的序列
        max_dissim_val = -1.0
        start_i, start_j = -1, -1
        for i in range(num_sequences):
            for j in range(i + 1, num_sequences):
                if dissimilarity_matrix[i, j] > max_dissim_val:
                    max_dissim_val = dissimilarity_matrix[i, j]
                    start_i, start_j = i, j

        if self.k < 2:
            return [0] if self.k == 1 else []
        if start_i == -1:  # 所有序列都完全相同
            return list(range(self.k))

        selected_indices = {start_i, start_j}

        while len(selected_indices) < self.k:
            max_min_dist = -1.0
            best_next_idx = -1
            candidate_indices = set(range(num_sequences)) - selected_indices

            for cand_idx in candidate_indices:
                min_dist_to_selected = min(
                    dissimilarity_matrix[cand_idx, sel_idx] for sel_idx in selected_indices  # type: ignore
                )
                if min_dist_to_selected > max_min_dist:
                    max_min_dist = min_dist_to_selected
                    best_next_idx = cand_idx

            if best_next_idx != -1:
                selected_indices.add(best_next_idx)
            else:  # 如果剩余候选项都完全相同，则随便选一个
                selected_indices.add(candidate_indices.pop())

        return list(selected_indices)


def test():
    filter = DiverseFilter(k=2)
    sequences = [
        [1, 1, 2, 2, 5, 0, 0, 0],
        [1, 1, 2, 2, 5, 6, 0, 0],
        [2, 2, 5, 1, 1, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7, 8],
        # [2, 1, 3, 4, 5, 6, 7, 9],
    ]
    start_time = time.time()
    indices = filter(torch.tensor(sequences), 0)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(sorted(indices))
    assert sorted(indices) in [[2, 3], [2, 4], [0, 4]]


if __name__ == "__main__":
    test()
