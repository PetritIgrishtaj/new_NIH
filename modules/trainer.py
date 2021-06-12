import os
import gc
import time
from typing import List

from .loss import AverageMeter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

if "TPU" in os.environ:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl


def get_roc_auc_score(y_true, y_probs, labels):
    class_roc_auc_list = dict()

    for i in range(y_true.shape[-1]):
        try:
            class_roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
            class_roc_auc_list[labels[i]] = class_roc_auc
        except:
            class_roc_auc_list[labels[i]] = None


    return class_roc_auc_list

def train_fn_tpu(
    model,
    epoch: int,
    para_loader,
    optimizer,
    criterion,
    scheduler,
    device
):
    # Model must be a global variable
    model.train()
    trn_loss_meter = AverageMeter()

    for batch_idx, (inputs, labels) in enumerate(para_loader):
        # extract inputs and labels
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # forward and backward pass
        preds = model(inputs)
        loss  = criterion(preds, labels)
        loss.backward()
        xm.optimizer_step(optimizer, barrier = True) # barrier is required on single-core training but can be dropped with multiple cores

        # compute loss
        trn_loss_meter.update(loss.detach().item(), inputs.size(0))

        # feedback
        if (batch_idx > 0) and (batch_idx % batch_verbose == 0):
            xm.master_print('-- batch {} | cur_loss = {:.6f}, avg_loss = {:.6f}'.format(
                batch_idx, loss.item(), trn_loss_meter.avg))

        # clear memory
        del inputs, labels, preds, loss
        gc.collect()

        # early stop
        if batch_idx > batches_per_epoch:
            break

    # scheduler step
    scheduler.step()

    # clear memory
    del para_loader, batch_idx
    gc.collect()

    return trn_loss_meter.avg

def valid_fn_tpu(model, epoch, para_loader, criterion, device):

    # initialize
    model.eval()
    val_loss_meter = AverageMeter()

    # validation loop
    for batch_idx, (inputs, labels) in enumerate(para_loader):

        # extract inputs and labels
        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute preds
        with torch.no_grad():
            preds = model(inputs)
            loss  = criterion(preds, labels)

        # compute loss
        val_loss_meter.update(loss.detach().item(), inputs.size(0))

        # feedback
        if (batch_idx > 0) and (batch_idx % batch_verbose == 0):
            xm.master_print('-- batch {} | cur_loss = {:.6f}, avg_loss =  {:.6f}'.format(
                batch_idx, loss.item(), val_loss_meter.avg))

        # clear memory
        del inputs, labels, preds, loss
        gc.collect()

    # clear memory
    del para_loader, batch_idx
    gc.collect()

    return val_loss_meter.avg


def train_epoch(device, train_loader, model, loss_fn, optimizer, epochs_till_now, final_epoch, log_interval):
    '''
    Takes in the data from the 'train_loader', calculates the loss over it using the 'loss_fn'
    and optimizes the 'model' using the 'optimizer'

    Also prints the loss and the ROC AUC score for the batches, after every 'log_interval' batches.
    '''
    model.train()

    running_train_loss = 0
    train_loss_list = []

    start_time = time.time()
    for batch_idx, (img, target) in enumerate(train_loader):

        img, target = img.to(device), target.to(device)

        optimizer.zero_grad()
        out = model(img)
        loss = loss_fn(out, target)
        running_train_loss += loss.item()*img.shape[0]
        train_loss_list.append(loss.item())

        loss.backward()
        optimizer.step()

        if (batch_idx+1)%log_interval == 0:
            batch_time = time.time() - start_time
            m, s = divmod(batch_time, 60)
            print('Train Loss for batch {}/{} @epoch{}/{}: {} in {} mins {} secs'.format(
                str(batch_idx+1).zfill(3),
                str(len(train_loader)).zfill(3),
                epochs_till_now,
                final_epoch,
                round(loss.item(), 5),
                int(m),
                round(s, 2)
            ))

        start_time = time.time()

    return train_loss_list, running_train_loss/float(len(train_loader.dataset))

def val_epoch(device, val_loader, model, loss_fn, labels, epochs_till_now = None,
              final_epoch = None, log_interval = 1, test_only = False):
    '''
    It essentially takes in the val_loader/test_loader, the model and the loss function and evaluates
    the loss and the ROC AUC score for all the data in the dataloader.

    It also prints the loss and the ROC AUC score for every 'log_interval'th batch, only when 'test_only' is False
    '''
    model.eval()

    running_val_loss = 0
    val_loss_list = []
    val_loader_examples_num = len(val_loader.dataset)

    probs = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
    gt    = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
    k=0

    with torch.no_grad():
        batch_start_time = time.time()
        for batch_idx, (img, target) in enumerate(val_loader):
            img = img.to(device)
            target = target.to(device)

            out = model(img)
            loss = loss_fn(out, target)
            running_val_loss += loss.item()*img.shape[0]
            val_loss_list.append(loss.item())

            # storing model predictions for metric evaluation
            probs[k: k + out.shape[0], :] = out.cpu()
            gt[   k: k + out.shape[0], :] = target.cpu()
            k += out.shape[0]

            if ((batch_idx+1)%log_interval == 0):

                batch_time = time.time() - batch_start_time
                m, s = divmod(batch_time, 60)
                print('Val Loss   for batch {}/{} @epoch{}/{}: {} in {} mins {} secs'.format(
                    str(batch_idx+1).zfill(3),
                    str(len(val_loader)).zfill(3),
                    epochs_till_now,
                    final_epoch,
                    round(loss.item(), 5),
                    int(m),
                    round(s, 2)
                ))

            batch_start_time = time.time()

    # metric scenes
    roc_auc = get_roc_auc_score(gt, probs, labels)

    return val_loss_list, running_val_loss/float(len(val_loader.dataset)), roc_auc

def run_tpu(
    model,
    train_dataset,
    valid_dataset,
    train_bs,
    valid_bs,
    learn_params,
    num_epochs,
    device
):
    eta, step, gamma = learn_params

    ### DATA PREP

    # data samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas = xm.xrt_world_size(),
                                                                    rank         = xm.get_ordinal(),
                                                                    shuffle      = True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,
                                                                    num_replicas = xm.xrt_world_size(),
                                                                    rank         = xm.get_ordinal(),
                                                                    shuffle      = False)

    # data loaders
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size  = valid_bs,
                                               sampler     = valid_sampler,
                                               num_workers = 0,
                                               pin_memory  = True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size  = train_bs,
                                               sampler     = train_sampler,
                                               num_workers = 0,
                                               pin_memory  = True)


    # scale LR
    scaled_eta = eta * xm.xrt_world_size()

    # optimizer and loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = scaled_eta)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = step, gamma = gamma)


    ### MODELING

    # placeholders
    trn_losses = []
    val_losses = []
    best_val_loss = 1

    # modeling loop
    gc.collect()
    for epoch in range(num_epochs):

        # display info
        xm.master_print('-'*55)
        xm.master_print('EPOCH {}/{}'.format(epoch + 1, num_epochs))
        xm.master_print('-'*55)
        xm.master_print('- initialization | TPU cores = {}, lr = {:.6f}'.format(
            xm.xrt_world_size(), scheduler.get_last_lr() / xm.xrt_world_size()))
        epoch_start = time.time()
        gc.collect()

        # update train_loader shuffling
        train_loader.sampler.set_epoch(epoch)

        # training pass
        train_start = time.time()
        xm.master_print('- training...')
        para_loader = pl.ParallelLoader(train_loader, [device])
        trn_loss = train_fn_tpu(model       = model,
                                epoch       = epoch + 1,
                                para_loader = para_loader.per_device_loader(device),
                                criterion   = criterion,
                                optimizer   = optimizer,
                                scheduler   = scheduler,
                                device      = device)
        del para_loader
        gc.collect()

        # validation pass
        valid_start = time.time()
        xm.master_print('- validation...')
        para_loader = pl.ParallelLoader(valid_loader, [device])
        val_loss = valid_fn_tpu(model       = model,
                                epoch       = epoch + 1,
                                para_loader = para_loader.per_device_loader(device),
                                criterion   = criterion,
                                device      = device)
        del para_loader
        gc.collect()

        # save weights
        if val_loss < best_val_loss:
            xm.save(model.state_dict(), 'weights_{}.pt'.format(model_name))
            best_val_loss = val_loss

        # display info
        xm.master_print('- elapsed time | train = {:.2f} min, valid = {:.2f} min'.format(
            (valid_start - train_start) / 60, (time.time() - valid_start) / 60))
        xm.master_print('- average loss | train = {:.6f}, valid = {:.6f}'.format(
            trn_loss, val_loss))
        xm.master_print('-'*55)
        xm.master_print('')

        # save losses
        trn_losses.append(trn_loss)
        val_losses.append(val_loss)
        del trn_loss, val_loss
        gc.collect()

    # print results
    xm.master_print('Best results: loss = {:.6f} (epoch {})'.format(np.min(val_losses), np.argmin(val_losses) + 1))

    return trn_losses, val_losses

def run(device: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model: Module,
        epochs: int,
        loss_fn: _Loss,
        optimizer: Optimizer,
        log_interval: int,
        save_interval: int,
        labels: List,
        lr: float,
        model_dir: str):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model.to(device)
    loss_fn.to(device)

    for epoch in range(1, epochs+1):
        print('--- TRAIN ---')
        train_epoch(device=device,
                    train_loader=train_loader,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    epochs_till_now=epoch,
                    final_epoch=epochs,
                    log_interval=log_interval)

        print('--- VAL ---')
        _, _, roc = val_epoch(device=device,
                              val_loader=val_loader,
                              model=model,
                              loss_fn=loss_fn,
                              labels=labels,
                              epochs_till_now=epoch,
                              final_epoch=epochs,
                              log_interval=log_interval)
        print('ROC_AUC_SCORE: {}'.format(roc))

        if (epoch%save_interval == 0):
            model_loc = os.path.join(model_dir, 'model_weights_epoch_{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_loc)
            print('Model saved to {}'.format(model_loc))


    model_loc = os.path.join(model_dir, 'model_weights_final.pth')
    torch.save(model.state_dict(), model_loc)
    print('Model saved to {}'.format(model_loc))
