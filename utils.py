import argparse


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type = str, default = "checkpoint/checkpoint-2994")
    args = parser.parse_args()

    return args