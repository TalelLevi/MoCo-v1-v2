import os
import argparse


def parse_cli(*args, **kwargs):
    def is_dir(dirname):
        if not os.path.isdir(dirname):
            raise argparse.ArgumentTypeError(f"{dirname} is not a directory")
        else:
            return dirname

    def is_file(filename):
        if not os.path.isfile(filename):
            raise argparse.ArgumentTypeError(f"{filename} is not a file")
        else:
            return filename

    if 'hello' in kwargs.keys():
        print(kwargs['hello'])

    if 'bye' in kwargs.keys():
        print(kwargs['bye'])
    # parser = argparse.ArgumentParser(description="parse parameters for the model")
    # sp = parser.add_subparsers(help="Sub-command help")



    # args = parser.parse_args()

    # return args

parse_cli(hello = 1, bye = 2)
# or
parse_cli(**{'hello': 2, 'bye': 4})
