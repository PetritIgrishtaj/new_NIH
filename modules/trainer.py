import os
import time
from typing import List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


def get_roc_auc_score(y_true, y_probs, labels):
    class_roc_auc_list = dict()

    for i in range(y_true.shape[-1]):
        try:
            class_roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
            class_roc_auc_list[labels[i]] = class_roc_auc
        except:
            class_roc_auc_list[labels[i]] = None


    return class_roc_auc_list


def train_epoch(device, train_loader, model, loss_fn, optimizer,
                epochs_till_now, final_epoch, scheduler, log_interval):
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

    # scheduler step
    scheduler.step()

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

def run(device: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        scheduler,
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
                    scheduler=scheduler,
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
