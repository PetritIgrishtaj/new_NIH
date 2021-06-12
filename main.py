import argparse
import os
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchvision import transforms

from modules import net, trainer
from modules.dataset import ChestXRayImageDataset, ChestXRayImages
from typing import List, Callable, Optional

if "TPU" in os.environ:
    # XLA imports
    import torch_xla
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.data_parallel as dp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.test.test_utils as test_utils



transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def setup_datasets(
    data_path: str,
    folds: int = 5,
    fold_id: int = 0,
    frac: float = 1.,
    transform: Optional[List[Callable]] = None
):
    data_wrapper = ChestXRayImages(data_path, folds=folds, frac=frac)
    data_train = ChestXRayImageDataset(
        data_path,
        data_wrapper.data_train(fold_id),
        transform=transform
    )
    data_val = ChestXRayImageDataset(
        data_path,
        data_wrapper.data_val(fold_id),
        transform=transform
    )
    data_test = ChestXRayImageDataset(
        data_path,
        data_wrapper.data_test,
        transform=transform
    )

    return data_train, data_val, data_test


# wrapper for multi core processing
def get_mp_wrapper(model):
    def _mp_wrapper(rank, flags)
        torch.set_default_tensor_type('torch.FloatTensor')
        trn_losses, val_losses = trainer.run_tpu(model)
        np.save('trn_losses.npy', np.array(trn_losses))
        np.save('val_losses.npy', np.array(val_losses))

    return _mp_wrapper



def main():
    # modeling
    batch_size        = 128  # num_images = batch_size*num_tpu_workers
    batches_per_epoch = 1000 # num_images = batch_size*batches_per_epoch*num_tpu_workers
    num_epochs        = 1
    batch_verbose     = 100
    num_tpu_workers   = 8

    # learning rate
    eta   = 0.0001
    step  = 1
    gamma = 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type = str, help = 'Path to training data')
    parser.add_argument('--model-path', type = str, help = 'Path to store models')
    parser.add_argument('--test-bs', type = int, default = 64, help = 'test batch size')
    parser.add_argument('--val-bs', type = int, default = 64, help = 'val batch size')
    parser.add_argument('--train-bs', type = int, default = 64, help = 'train batch size')
    parser.add_argument('--lr', type = float, default = 1e-5, help = 'Learning Rate passed to optimizer')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Force usage of device')
    parser.add_argument('--epochs', type = str, default = 2, help = 'Train for n epochs')
    parser.add_argument('--log-interval', type = int, default = 5, help = 'log every n batches')
    parser.add_argument('--save-interval', type = int, default = 5, help = 'save every n batches')
    parser.add_argument('--data-frac', type = float, default = 1, help = 'use only fraction of the data')
    parser.add_argument('--folds', type=int, default=5, help='how many folds to produce')
    parser.add_argument('--fold-id', type=int, default=0, help='Which fold id to use for test/val split')
    parser.add_argument('--seed', type=int, default=0, help='Seed the random generator to get reproducability')
    args = parser.parse_args()

    # Initialize the data sets
    data_train, data_val, data_test = setup_datasets(args.data_path,
                                                     folds=args.folds,
                                                     fold_id=args.fold_id,
                                                     frac=args.data_frac,
                                                     transform=transform)

    # Initialize model
    model = net.get_model(len(ChestXRayImageDataset.labels))

    # device specific setup
    if args.device == 'tpu':
        mx = xmp.MpModelWrapper(g.model)
        device = xm.xla_device()
        model  = mx.to(device)
    elif args.device == 'cuda':
        device = torch.device("cuda")
        model.to(device)
    else:
        device = torch.device('cpu')
        model.to(device)

    # modeling
    gc.collect()
    FLAGS = {}
    xmp.spawn(get_mp_wrapper(model), args = (FLAGS,), nprocs = num_tpu_workers, start_method = 'fork')

if __name__ == "__main__":
    main()
