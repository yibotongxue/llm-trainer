import argparse

from llm_trainer.trainer.iterative_sft import IterativeSftTrainer
from llm_trainer.utils.config import (
    deepcopy_config,
    load_config,
    update_config_with_unparsed_args,
)

# import deepspeed


# deepspeed.ops.op_builder.CPUAdamBuilder().load()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-file-path",
        type=str,
        required=True,
        help="The path to the config file",
    )
    args, unparsed_args = parser.parse_known_args()

    cfgs = load_config(args.config_file_path)
    update_config_with_unparsed_args(unparsed_args=unparsed_args, cfgs=cfgs)

    cfgs = deepcopy_config(cfgs)

    cfgs.pop("_common")

    trainer = IterativeSftTrainer(**cfgs)

    trainer.train()


if __name__ == "__main__":
    main()
