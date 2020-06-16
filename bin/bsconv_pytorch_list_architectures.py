#!/usr/bin/env python3

import argparse
import sys

import bsconv.pytorch.provider


def get_args():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='List available BSConv PyTorch model architectures.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filter', nargs='*', type=str, help='Filter string(s). Only architectures which include all of the provided strings are shown. Leave unset to show all architectures.')
    return parser.parse_args()


def main():
    args = get_args()
    model_names = sorted(bsconv.pytorch.provider.models.keys())
    for model_name in model_names:
        if all(filter_ in model_name for filter_ in args.filter):
            print(model_name)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped')
        sys.exit(0)
    except Exception as e:
        print('Error: {}'.format(e))
        sys.exit(1)
