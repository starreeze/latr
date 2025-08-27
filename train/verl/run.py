import os
import sys

import yaml

import verl
import verl.trainer.main_ppo
from verl.workers.rollout.kt_rollout import kt_conf_path

assert verl.__version__ == "0.5.0"


def handle_args():
    kt_idx: list[int] = []
    kt_args: list[str] = []
    for i, arg in enumerate(sys.argv):
        if arg.startswith("kt."):
            kt_idx.append(i)
            kt_args.append(arg[3:])

    if not kt_idx:
        print("No KeyTokenGenConfig found, using default model")

    kwargs = {}
    for arg in kt_args:
        k, v = arg.split("=", 1)
        kwargs[k.strip()] = v.strip()

    os.makedirs(os.path.dirname(kt_conf_path), exist_ok=True)
    yaml.dump(kwargs, open(kt_conf_path, "w"))

    kt_idx.reverse()
    for idx in kt_idx:
        sys.argv.pop(idx)


def main():
    handle_args()
    verl.trainer.main_ppo.main()  # type: ignore


if __name__ == "__main__":
    main()
