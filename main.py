import argparse
from tools.config import DictToAttributes, yaml_model_load
from cvalgorithm.trainer import TrainerManager


def parse_args() -> object:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--base', default='./config/default.yaml', help='')
    parser.add_argument('--config', default='./config/segment/baseline.yaml', help='model config')
    parser.add_argument("--ds", "--d", default=None, help="s3 address co")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.ds is not None:
        pass
    type_config = yaml_model_load(args.config)
    trainer = TrainerManager(type_config)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return


if __name__ == '__main__':
    main()
