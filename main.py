import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchvision import transforms

from modules import net, trainer, loss
from modules.dataset import ChestXRayImageDataset, ChestXRayImages


transform = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type = str, help = 'Path to training data')
    parser.add_argument('--model-path', type = str, help = 'Path to store models')
    parser.add_argument('--test-bs', type = int, default = 64, help = 'test batch size')
    parser.add_argument('--val-bs', type = int, default = 64, help = 'val batch size')
    parser.add_argument('--train-bs', type = int, default = 64, help = 'train batch size')
    parser.add_argument('--lr', type = float, default = 0.0001, help = 'Learning Rate passed to optimizer')
    parser.add_argument('--step', type = int, default = 1)
    parser.add_argument('--gamma', type = float, default = 0.5)
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Force usage of device')
    parser.add_argument('--epochs', type = int, default = 2, help = 'Train for n epochs')
    parser.add_argument('--log-interval', type = int, default = 5, help = 'log every n batches')
    parser.add_argument('--save-interval', type = int, default = 5, help = 'save every n batches')
    parser.add_argument('--data-frac', type = float, default = 1, help = 'use only fraction of the data')
    parser.add_argument('--folds', type=int, default=5, help='how many folds to produce')
    parser.add_argument('--fold-id', type=int, default=0, help='Which fold id to use for test/val split')
    parser.add_argument('--seed', type=int, default=0, help='Seed the random generator to get reproducability')
    args = parser.parse_args()

    if args.device == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')


    data_wrapper = ChestXRayImages(root  = args.data_path,
                                   folds = args.folds,
                                   frac  = args.data_frac,
                                   seed  = args.seed)

    data_train = ChestXRayImageDataset(
        args.data_path,
        data_wrapper.data_train(args.fold_id),
        transform=transform
    )
    data_val = ChestXRayImageDataset(
        args.data_path,
        data_wrapper.data_val(args.fold_id),
        transform=transform
    )
    data_test = ChestXRayImageDataset(
        args.data_path,
        data_wrapper.data_test,
        transform=transform
    )


    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.test_bs)
    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=args.val_bs)
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.train_bs)

    model = net.get_model(len(ChestXRayImageDataset.labels))

    # Print Network and training info
    summary(model, input_size=(args.train_bs, 3, 244, 244))
    print('Using device: {}'.format(device))
    print('With {} Test datasets, {} val data sets and {} train datasets'.format(
        len(data_test), len(data_val), len(data_train)
    ))

    # Setting the training variables
    trainer.criterion_t = nn.BCEWithLogitsLoss()
    trainer.criterion_v = nn.BCEWithLogitsLoss()
    # trainer.criterion_t = loss.BPMLLLoss()
    # trainer.criterion_v = loss.BPMLLLoss()
    trainer.optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr
    )
    trainer.scheduler = optim.lr_scheduler.StepLR(
        trainer.optimizer,
        step_size = args.step,
        gamma = args.gamma
    )
    # trainer.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     trainer.optimizer,
    #     factor=0.5,
    #     patience=2
    # )

    # Run the training
    trainer.run(device        = device,
                model         = model,
                train_loader  = train_loader,
                val_loader    = val_loader,
                epochs        = args.epochs,
                log_interval  = args.log_interval,
                save_interval = args.save_interval,
                labels        = data_train.labels,
                model_dir     = args.model_path,
                stage         = '0')

    # reset trainable parameters.
    # possibly adjust optimizer and scheduler
    # rerun trainer.run()

if __name__ == "__main__":
    main()
