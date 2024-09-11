import argparse
from tools.config import DictToAttributes, yaml_model_load, CfgNode, get_cfg
from cvalgorithm.trainer import TrainerManager


def parse_args() -> object:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--base', default='./config/default.yaml', help='')
    parser.add_argument('--config', default='./config/segment/baseline.yaml', help='model config')
    parser.add_argument("--ds", "--d", default=None, help="s3 address co")
    args = parser.parse_args()
    return args


def main():
    cfg = get_cfg()
    args = parse_args()
    cfg.merge_from_file(args.config)
    if args.ds is not None:
        pass

    trainer = TrainerManager(cfg)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return


if __name__ == '__main__':
    main()