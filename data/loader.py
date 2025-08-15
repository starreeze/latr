from datasets import Dataset

from args import DataArgs
from data import sft
from data.base import DataBase
from data.rl import RLData  # noqa: F401
from data.sft import SFTData  # noqa: F401


def load(
    data_type: str, data_dir: str, dataset_name: str, eval_dataset: str, **kwargs
) -> tuple[Dataset, Dataset]:
    data_cls: type[DataBase] = globals()[f"{data_type.upper()}Data"]
    train: Dataset = data_cls(data_dir, dataset_name, "train", **kwargs).load()
    test: Dataset = data_cls(data_dir, eval_dataset, "test", **kwargs).load()
    return train, test


def get_sft_collator(tokenizer, data_args: DataArgs, **kwargs):
    """Create a data collator for the given step."""
    sft_collator_cls = getattr(sft, f"{data_args.latent_type.capitalize()}SFTCollator")
    return sft_collator_cls(tokenizer, **data_args.__dict__, **kwargs)
