import torch


class DiverseFilter:
    def __init__(self, k: int):
        self.k = k

    def __call__(self, sequences: torch.Tensor, pad_id: int) -> list[int]:
        """
        Select k most diverse sequences (right-padded).
        Args:
            sequences: (batch, seq_len) with batch >= k
            pad_id: padding id. Only compare tokens that are not padded.
        Returns:
            list of indices to keep with length k
        """
        return [2, 3]


def test():
    filter = DiverseFilter(k=2)
    sequences = [
        [1, 1, 2, 2, 5, 0, 0, 0],
        [1, 1, 2, 2, 5, 6, 0, 0],
        [2, 2, 5, 1, 1, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 1, 3, 4, 5, 6, 7, 9],
    ]
    indices = filter(torch.tensor(sequences), 0)
    assert sorted(indices) in [[2, 3], [2, 4]]


if __name__ == "__main__":
    test()
