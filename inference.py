import os
import argparse
from tqdm import tqdm

import torch

import data as Dataset
from config import Config
from util.distributed import init_dist
from util.logging import init_logging, make_logging_dir
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/face.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--output_dir', type=str)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)

    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)    
        opt.device = torch.cuda.current_device()
    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    make_logging_dir(logdir, date_uid)

    # create a dataset
    val_dataset, train_dataset = Dataset.get_train_val_dataloader(opt.data)

    # create a model
    net_G, net_D, net_G_ema, opt_G, opt_D, sch_G, sch_D \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_D, net_G_ema, \
                          opt_G, opt_D, sch_G, sch_D, \
                          train_dataset, val_dataset)

    current_epoch, current_iteration = trainer.load_checkpoint(opt, args.which_iter)                          

    output_dir = os.path.join(
        args.output_dir, 
        'epoch_{:05}_iteration_{:09}'.format(current_epoch, current_iteration)
        )
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for data in tqdm(val_dataset):
            data = trainer.start_of_iteration(data, current_iteration)
            trainer.test(data, output_dir, current_iteration)

    print('Done')
