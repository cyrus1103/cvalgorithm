import argparse
from cvalgorithm.data_analysis.utils import EasyDict


def parse_args() -> object:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--config', help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    test_dict = dict(a=200)
    EasyDict(test_dict)


if __name__ == '__main__':
    main()
