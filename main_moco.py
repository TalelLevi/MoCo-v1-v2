import os
import random
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from src import utils
from config import cfg
from src import pytorch_utils as ptu
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from src.MoCo import MoCo_v2
import time
import warnings
warnings.filterwarnings("ignore")
from src.utils import *
#
# if cfg.pretraining.load is not None and os.path.exists(os.path.join(cfg.models_dir, cfg.pretraining.version, ptu.naming_scheme(cfg.pretraining.version, epoch=cfg.pretraining.load)) + '.pth'):
#     checkpoint = ptu.load_model(version=cfg.pretraining.version, models_dir=cfg.models_dir, epoch=cfg.pretraining.load)
#     if cfg.prints == 'display':
#         display(checkpoint.log.sort_index(ascending=False).head(20))
#     elif cfg.prints == 'print':
#         print(checkpoint.log.sort_index(ascending=False).head(20))
# else:
#     model = arch.MoCo_v2(backbone=cfg.pretraining.backbone,
#                          dim=cfg.pretraining.dim,
#                          queue_size=cfg.pretraining.queue_size,
#                          batch_size=cfg.pretraining.bs,
#                          momentum=cfg.pretraining.model_momentum,
#                          temperature=cfg.pretraining.temperature,
#                          bias=cfg.pretraining.bias,
#                          moco=True,
#                          clf_hyperparams=cfg.pretraining.clf_kwargs,
#                          seed=cfg.seed,
#                          mlp=cfg.pretraining.mlp,
#                          )
#
#     checkpoint = utils.MyCheckpoint(version=cfg.pretraining.version,
#                                     model=model,
#                                     optimizer=optimizer,
#                                     criterion=nn.CrossEntropyLoss().to(device),
#                                     score=utils.accuracy_score,
#                                     lr_scheduler=lr_scheduler,
#                                     models_dir=cfg.models_dir,
#                                     seed=cfg.seed,
#                                     best_policy=cfg.pretraining.best_policy,
#                                     save=cfg.save,
#                                     )
#     if cfg.save:
#         with open(os.path.join(checkpoint.version_dir, 'config.txt'), 'w') as f:
#             f.writelines(str(cfg))
#
# ptu.params(checkpoint.model)
#
#
# # In[7]:
#
#
# train_dataset = utils.Dataset(os.path.join(cfg.data_path, 'train'), cfg.pretraining.train_transforms, preload_data=cfg.preload_data, tqdm_bar=cfg.tqdm_bar)
# train_eval_dataset = utils.Dataset(os.path.join(cfg.data_path, 'train'), cfg.pretraining.train_eval_transforms, preload_data=cfg.preload_data, tqdm_bar=cfg.tqdm_bar)
# val_dataset = utils.Dataset(os.path.join(cfg.data_path, 'val'), cfg.pretraining.val_eval_transforms, preload_data=cfg.preload_data, tqdm_bar=cfg.tqdm_bar)
#
# train_loader = torch.utils.data.DataLoader(train_dataset,
#                                            batch_size=checkpoint.model.batch_size,
#                                            num_workers=cfg.num_workers,
#                                            drop_last=True, shuffle=True, pin_memory=True)
#
# train_eval_loader = torch.utils.data.DataLoader(train_eval_dataset,
#                                                 batch_size=checkpoint.model.batch_size,
#                                                 num_workers=cfg.num_workers,
#                                                 drop_last=True, shuffle=True, pin_memory=True)
#
# val_loader = torch.utils.data.DataLoader(val_dataset,
#                                          batch_size=checkpoint.model.batch_size,
#                                          num_workers=cfg.num_workers,
#                                          drop_last=True, shuffle=False, pin_memory=True)
#
#
# # In[ ]:
#
#
# checkpoint.train(train_loader=train_loader,
#                  train_eval_loader=train_eval_loader,
#                  val_loader=val_loader,
#                  train_epochs=int(max(0, cfg.pretraining.epochs - checkpoint.get_log())),
#                  optimizer_params=cfg.pretraining.optimizer_params,
#                  prints=cfg.prints,
#                  epochs_save=cfg.epochs_save,
#                  epochs_evaluate_train=cfg.epochs_evaluate_train,
#                  epochs_evaluate_validation=cfg.epochs_evaluate_validation,
#                  device=device,
#                  tqdm_bar=cfg.tqdm_bar,
#                  save=cfg.save,
#                  save_log=cfg.save_log,
#                  )


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--data_path', default=os.path.join('data', 'imagenette2'), type=str)
    parser.add_argument('--models_dir', default='models', type=str)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--save_log', default=True, type=bool)
    parser.add_argument('--epochs_evaluate_train', default=1, type=int)
    parser.add_argument('--epochs_evaluate_validation', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--epochs_save', default=None, type=int)
    parser.add_argument('--tqdm_bar', default=True, type=bool)
    parser.add_argument('--preload_data', default=True, type=bool)
    parser.add_argument('--prints', default='print', type=str)

    # * MoCo
    parser.add_argument('--load', default=-1, type=int)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--backbone', default='resnext50_32x4d', type=str)
    parser.add_argument('--bs', default=32, type=int)
    parser.add_argument('--temperature', default=0.2, type=float)
    parser.add_argument('--queue_size', default=16384, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--optimizer_momentum', default=0.9, type=float)
    parser.add_argument('--lr', default=3e-2, type=float)
    parser.add_argument('--min_lr', default=5e-7, type=float)
    parser.add_argument('--cos', default=True, type=bool)
    parser.add_argument('--best_policy', default='val_score', type=str)
    parser.add_argument('--model_momentum', default=0.999, type=float)
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--mlp', default=True, type=bool)
    parser.add_argument('--bias', default=True, type=bool)

    # clf params
    parser.add_argument('--clf_load', default=-1, type=int)
    parser.add_argument('--clf_moco_epoch', default='best', type=str)
    parser.add_argument('--clf_epochs', default=200, type=int)
    parser.add_argument('--clf_wd ', default= 0.0, type=float)
    parser.add_argument('--clf_lr', default=3e-2, type=float)
    parser.add_argument('--clf_cos', default=True, type=bool)
    parser.add_argument('--clf_best_policy', default= 'val_score', type=str)
    parser.add_argument('--clf_bs ', default= 32, type=int)
    parser.add_argument('--clf_optimizer_momentum', default=0.9, type=float)
    parser.add_argument('--clf_min_lr', default=5e-7, type=float)

    return parser

def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.load:
        pass
    else:
        # exp_name = create_name(args)
        exp_name='temp'

    model = MoCo_v2(backbone=args.backbone,
                    dim=args.dim,
                    queue_size=args.queue_size,
                    batch_size=args.bs,
                    momentum=args.model_momentum,
                    temperature=args.temperature,
                    bias=args.bias,
                    pretraining=True,
                    clf_hyperparams={'random_state': 42, 'max_iter': 10000},
                    seed=args.seed,
                    mlp=args.mlp
                    )
    if len(os.environ["CUDA_VISIBLE_DEVICES"])>1:
        model = nn.DataParallel(model)
    model = model.to(device)

    train_dataset = utils.Dataset(os.path.join(args.data_path, 'train'), moco_v2_transforms,
                                  preload_data=args.preload_data, tqdm_bar=args.tqdm_bar)
    train_eval_dataset = utils.Dataset(os.path.join(args.data_path, 'train'), TwoCropsTransform(clf_train_transforms),
                                       preload_data=args.preload_data, tqdm_bar=args.tqdm_bar)
    val_dataset = utils.Dataset(os.path.join(args.data_path, 'val'), TwoCropsTransform(clf_val_transforms),
                                preload_data=args.preload_data, tqdm_bar=args.tqdm_bar)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.bs,
                                               num_workers=args.num_workers,
                                               drop_last=True, shuffle=True, pin_memory=True)

    train_eval_loader = torch.utils.data.DataLoader(train_eval_dataset,
                                                    batch_size=args.bs,
                                                    num_workers=args.num_workers,
                                                    drop_last=True, shuffle=True, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.bs,
                                             num_workers=args.num_workers,
                                             drop_last=True, shuffle=False, pin_memory=True)

    # ToDo : consider to remove it
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                           T_max=args.epochs,
    #                                                           eta_min=args.min_lr) if args.cos else None

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr,
                                momentum=args.optimizer_momentum,
                                weight_decay=args.wd)


    train.train(model,criterion,optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MoCo training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)




