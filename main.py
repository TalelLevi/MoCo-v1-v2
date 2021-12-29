import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import argparse
import shutil
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
from src import utils


warnings.filterwarnings("ignore")
from src.utils import *
from src.trainers.train import TorchTrainer as Trainer
from src.utils.parser import get_args_parser
from src.models.MoCo import MoCo_v2
from src.loss_functions.MoCo_loss import ContrastiveLoss

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
        print("hey")
    model = model.to(device)

    train_dataset = utils.Dataset(os.path.join(args.data_path, 'train'), moco_v2_transforms,
                                  preload_data=args.preload_data, tqdm_bar=args.tqdm_bar)
    # train_eval_dataset = utils.Dataset(os.path.join(args.data_path, 'train'), TwoCropsTransform(clf_train_transforms),
    #                                    preload_data=args.preload_data, tqdm_bar=args.tqdm_bar)
    val_dataset = utils.Dataset(os.path.join(args.data_path, 'val'), TwoCropsTransform(clf_val_transforms),
                                preload_data=args.preload_data, tqdm_bar=args.tqdm_bar)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.bs,
                                               num_workers=args.num_workers,
                                               drop_last=True, shuffle=True, pin_memory=True)

    # ToDo: consider to remove it
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=args.bs,
                                                    num_workers=args.num_workers,
                                                    drop_last=True, shuffle=True, pin_memory=True)


    # ToDo : consider to remove it
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                           T_max=args.epochs,
    #                                                           eta_min=args.min_lr) if args.cos else None

    criterion = ContrastiveLoss(pretraining=True)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr,
                                momentum=args.optimizer_momentum,
                                weight_decay=args.wd)

    if not args.load:
        if os.path.exists(f'./experiments/{exp_name}_moco'):
            shutil.rmtree(f'./experiments/{exp_name}_moco')

        Path(f'./experiments/{exp_name}_moco/checkpoints').mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, criterion, optimizer, device)
    res = trainer.fit(train_loader,val_loader,args.epochs,checkpoint_path=f'./experiments/{exp_name}_moco/checkpoints/model.pth')

    for y_axis, name in zip(res[1:], ['train_loss' , 'train_acc', 'test_loss', 'test_acc']):  # TODO change to plotter
        plt.plot(y_axis, label=name)
        plt.savefig(f'./plot_{name}.jpg')
        plt.clf()











    ##################### PART 2 #############################



    model = torch.load(os.path.join(args.load_path, 'model.pth'), map_location='cpu').module
    model.end_moco_phase()
    # model.module.pretraining = False

    # if config.checkpoint_path is not None:
    #     model_state_dict = torch.load(config.checkpoint_path, map_location='cpu')
    #     if 'model_state_dict' in model_state_dict.keys():
    #         model_state_dict = model_state_dict['model_state_dict']
    #     model_without_ddp.load_state_dict(model_state_dict, strict=True)

    if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1:
        model = nn.DataParallel(model)
    model = model.to(device)


    train_dataset = utils.Dataset(os.path.join(args.data_path, 'train'), utils.clf_train_transforms,
                                  preload_data=args.preload_data, tqdm_bar=args.tqdm_bar)
    # train_eval_dataset = utils.Dataset(os.path.join(args.data_path, 'train'), TwoCropsTransform(clf_train_transforms),
    #                                    preload_data=args.preload_data, tqdm_bar=args.tqdm_bar)
    val_dataset = utils.Dataset(os.path.join(args.data_path, 'val'), utils.clf_val_transforms,
                                preload_data=args.preload_data, tqdm_bar=args.tqdm_bar)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.bs,
                                               num_workers=args.num_workers,
                                               drop_last=True, shuffle=True, pin_memory=True)

    # train_eval_loader = torch.utils.data.DataLoader(train_eval_dataset,
    #                                                 batch_size=args.bs,
    #                                                 num_workers=args.num_workers,
    #                                                 drop_last=True, shuffle=True, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.bs,
                                             num_workers=args.num_workers,
                                             drop_last=True, shuffle=False, pin_memory=True)

    # ToDo : consider to remove it
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                           T_max=args.epochs,
    #                                                           eta_min=args.min_lr) if args.cos else None

    criterion = ContrastiveLoss(pretraining=False)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.clf_lr,
                                momentum=args.clf_optimizer_momentum,
                                weight_decay=args.clf_wd)

    if not args.load:
        if os.path.exists(f'./experiments/{exp_name}_clf'):
            shutil.rmtree(f'./experiments/{exp_name}_clf')

        Path(f'./experiments/{exp_name}_clf/checkpoints').mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, criterion, optimizer, device)
    res = trainer.fit(train_loader, val_loader, args.epochs,
                      checkpoint_path=f'./experiments/{exp_name}_clf/checkpoints/model.pth')
    for y_axis, name in zip(res[1:], ['train_loss', 'train_acc', 'test_loss', 'test_acc']): # TODO change to plotter
        plt.plot(y_axis, label=name)
        plt.savefig(f'./plot_{name}_clf.jpg')
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MoCo training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)



