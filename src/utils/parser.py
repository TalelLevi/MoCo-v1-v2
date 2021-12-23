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

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    if 'hello' in kwargs.keys():
        print(kwargs['hello'])

    if 'bye' in kwargs.keys():
        print(kwargs['bye'])
    # parser = argparse.ArgumentParser(description="parse parameters for the model")
    # sp = parser.add_subparsers(help="Sub-command help")

    # ========================== General model params ========================== #
    parser.add_argument(
        '--seed',
        default=None,
        type=int
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        default=3e-2,
        type=float
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        default=32,
        type=int
    )
    parser.add_argument(
        '--epochs',
        default=600,
        type=int,
        help="maximum number of epochs"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=20,
        help="Stop after this many epochs without any improvement",
    )
    parser.add_argument(
        '--backbone',
        default='resnext50_32x4d',
        type=str,
        help="Pick a backbone from Torch backbones"
    )
    # new
    parser.add_argument(
        '-loss'
        '--loss_function',
        default=None,      # TODO add torch crossentropy
        type=bool
    )
    parser.add_argument(
        '-opt'
        '--optimizer',
        default=None,      # TODO add torch adam
        type=bool
    )

    # ========================== General utility params ========================== #
    parser.add_argument(
        '--tqdm_bar',
        default=True,
        type=bool
    )
    parser.add_argument(
        '--num_workers',
        default=12,
        type=int
    )
    parser.add_argument(
        '--ml_ops',
        default=False,
        type=bool
    )

    # ========================== specific loss params ========================== #
    parser.add_argument('--temperature', default=0.2, type=float)
    parser.add_argument('--min_lr', default=5e-7, type=float)

    # ========================== Loading Saving & auditing ========================== #
    parser.add_argument('--save_checkpoints', default=True, type=bool)
    parser.add_argument('--checkpoints_path', default='checkpoints', type=str)

    parser.add_argument('--save_model', default=True, type=bool)
    parser.add_argument('--load_model', default=False, type=bool)
    parser.add_argument('--models_path', default='models', type=str)

    parser.add_argument('--save_log', default=False, type=bool)
    parser.add_argument('--logs_path', default='logs', type=str)

    parser.add_argument('--preload_data', default=True, type=bool)
    parser.add_argument('--data_path', default=os.path.join('data', 'imagenette2'), type=str)





    # ========================== project specific ========================== #

    parser.add_argument('--epochs_evaluate_train', default=1, type=int)
    parser.add_argument('--epochs_evaluate_validation', default=1, type=int)
    parser.add_argument('--epochs_save', default=None, type=int)
    parser.add_argument('--prints', default='print', type=str)
    parser.add_argument('--save', default=True, type=bool)



    # * MoCo
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--queue_size', default=16384, type=int)
    parser.add_argument('--optimizer_momentum', default=0.9, type=float)
    parser.add_argument('--cos', default=True, type=bool)
    parser.add_argument('--best_policy', default='val_score', type=str)
    parser.add_argument('--model_momentum', default=0.999, type=float)
    parser.add_argument('-dim', '--latent_dimension', default=128, type=int)
    parser.add_argument('--mlp', default=True, type=bool)
    parser.add_argument('--bias', default=True, type=bool)

    # clf params
    parser.add_argument('--clf_moco_epoch', default='best', type=str)
    parser.add_argument('--clf_epochs', default=200, type=int)
    parser.add_argument('--clf_best_policy', default='val_score', type=str)
    parser.add_argument('--clf_optimizer_momentum', default=0.9, type=float)


    args = parser.parse_args()

    return args

# example:
# parse_cli(hello = 1, bye = 2)
#            or
# parse_cli(**{'hello': 2, 'bye': 4})
