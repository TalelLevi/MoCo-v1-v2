import os

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from src import utils
import shutil


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from src.MoCo import MoCo_v2
import time
import warnings
warnings.filterwarnings("ignore")
from src.utils import *
from train import TorchTrainer as Trainer
import shutil


checkpoint.train(train_loader=train_loader,
                 val_loader=val_loader,
                 train_epochs=int(max(0, cfg.clf.epochs - checkpoint.get_log())),
                 optimizer_params=cfg.clf.optimizer_params,
                 prints=cfg.prints,
                 epochs_save=cfg.epochs_save,
                 epochs_evaluate_train=cfg.epochs_evaluate_train,
                 epochs_evaluate_validation=cfg.epochs_evaluate_validation,
                 device=device,
                 tqdm_bar=cfg.tqdm_bar,
                 save=cfg.save,
                 save_log=cfg.save_log,
                )


# In[ ]:


# import torchviz
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=0, drop_last=True, shuffle=True, pin_memory=True)
# for batch in train_loader:
#     img, labels = batch
#     img = img.to(device)
#     labels = labels.to(device)
#     model = model.to(device)
#     out = model(img, prints=True)
#     print('img',  img.shape)
#     print('labels', labels.shape)
#     print('out', out.shape)
#     loss = nn.functional.cross_entropy(out.float(), labels.long())
#     print('loss',  loss)
#     break
# torchviz.make_dot(out, params=dict(model.named_parameters()))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



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
    parser.add_argument('--load', default=False, type=bool)
    parser.add_argument('--load_path', default='./experiments/temp/checkpoints/', type=str)
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

    model = torch.load(args.load_path, map_location='cpu')
    model.pretraining = False

    if len(os.environ["CUDA_VISIBLE_DEVICES"])>1:
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

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.clf_lr,
                                momentum=args.clf_optimizer_momentum,
                                weight_decay=args.clf_wd)

    if not args.load:
        shutil.rmtree(f'./experiments/{exp_name}')

        Path(f'./experiments/{exp_name}/checkpoints').mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader,val_loader,args.epochs,checkpoint_path=f'./experiments/{exp_name}_clf/checkpoints/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MoCo training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)





