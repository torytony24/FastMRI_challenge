import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import copy
import os

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.teacher_varnet import Teacher_VarNet
from transformers import get_cosine_schedule_with_warmup


# Debugger: OFF 0 / ON 1
debugger = 0


def train_epoch(args, epoch, model, data_loader, optimizer, scheduler, loss_type, train_identifier):
    model.train()
    len_loader = len(data_loader)
    total_loss = 0.

    cnt=0
    for iter, data in enumerate(data_loader):
        cnt+=debugger
        if cnt>5:
            break
            
        mask, kspace, target, maximum, _, _, anatomy = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        anatomy = anatomy.cuda(non_blocking=True)

        if train_identifier != anatomy:
            continue
        
        output, _ = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} ',
            )
            
    total_loss = total_loss / len_loader
    return total_loss

def validate(args, model, data_loader, train_identifier):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)

    with torch.no_grad():
        cnt=0
        for iter, data in enumerate(data_loader):
            if cnt>5:
                break
                
            cnt+=debugger
            mask, kspace, target, _, fnames, slices, anatomy = data
            mask = mask.cuda(non_blocking=True)
            kspace = kspace.cuda(non_blocking=True)
            anatomy = anatomy.cuda(non_blocking=True)

            if train_identifier != anatomy:
                continue
            
            output, _ = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )

    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None
    


def save_model(model, best_val_loss, is_new_best, train_identifier):
    if train_identifier == 0:
        save_dir = '/root/FastMRI_challenge/Teacher_brain_savefile/checkpoints/'
    elif train_identifier == 1:
        save_dir = '/root/FastMRI_challenge/Teacher_knee_savefile/checkpoints/'
    torch.save(
        {
            'model': model.state_dict(),
            'best_val_loss': best_val_loss,
        },
        f=save_dir + 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(save_dir+ 'model.pt', save_dir+ 'best_model.pt')

def save_model_backup(args, exp_dir, epoch, model, optimizer, scheduler, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')



def train_teacher(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())


    ###########################
    # teacher: brain 0 / knee 1
    train_identifier = 0
    ###########################







    
    ###########################
    ### Model training part ###
    model = Teacher_VarNet(num_cascades=args.cascade, chans=args.chans, sens_chans=args.sens_chans)
    model.to(device=device)
    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle = True, is_train = True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, is_train = False)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 1e-4)
    loss_type = SSIMLoss().to(device=device)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 10,
        num_training_steps = args.num_epochs
    )

    best_val_loss = 1.
    start_epoch = 0
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        args.current_epoch = epoch

        print("@@@@@@@@@@@@@@@@@@@@@ [Train Model] @@@@@@@@@@@@@@@@@@@@@")
        train_loss = train_epoch(args, epoch, model, train_loader, optimizer, scheduler, loss_type, train_identifier)
        
        print("@@@@@@@@@@@@@@@@@@@@@ [Validate Model] @@@@@@@@@@@@@@@@@@@@@")
        val_loss, num_subjects, reconstructions, targets, inputs = validate(args, model, val_loader, train_identifier)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(model, best_val_loss, is_new_best, train_identifier)
        print(val_loss_log)

        # Backup for learning
        save_model_backup(args, args.exp_dir, epoch + 1, model, optimizer, scheduler, best_val_loss, is_new_best)
        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@ New Record @@@@@@@@@@@@@@@@@@@@@")
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)

