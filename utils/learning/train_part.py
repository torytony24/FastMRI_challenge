import shutil
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
#from utils.model.varnet import VarNet
from utils.model.aspin_varnet import ASPIN_VarNet, AnatomyClassifier
from utils.model.soft_aspin import SoftASPIN_VarNet

import os


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _, anatomy = data    # Add anatomy
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        anatomy = anatomy.cuda(non_blocking=True)    # Add anatomy

        output, probs = model(kspace, mask, anatomy)
        loss = loss_type(output, target, maximum)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

        # classification debugging
        if iter % 1000 == 0:
            _, pred_label = probs.max(dim=1)
            print('True:', anatomy.cpu().numpy())
            print('Pred:', pred_label.cpu().numpy())
            print('Prob:', probs.detach().cpu().numpy())
 
            
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices, _ = data    # Add anatomy
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

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
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def train_epoch_cls(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    len_loader = len(data_loader)
    total_loss = 0.
    correct = 0
    total_num = 0

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _, anatomy = data    # Add anatomy
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        anatomy = anatomy.cuda(non_blocking=True)    # Add anatomy

        output, probs = model(kspace, mask, anatomy)
        log_probs = torch.log(probs + 1e-8)
        loss = loss_type(log_probs, anatomy)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred = probs.argmax(dim=1)
        correct += (pred == anatomy).sum().item()
        total_num += anatomy.size(0)

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
            )

            acc = correct / total_num
            print(f'[Train Classifier] {correct:3d} out of {total_num:3d}, Acc: {acc:.4f}')
            
    total_loss = total_loss / len_loader
    return total_loss


def validate_cls(args, model, data_loader, loss_type):
    model.eval()
    total_loss = 0.
    correct = 0
    total_num = 0

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, maximum, _, _, anatomy = data    # Add anatomy
            mask = mask.cuda(non_blocking=True)
            kspace = kspace.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            anatomy = anatomy.cuda(non_blocking=True)    # Add anatomy
    
            output, probs = model(kspace, mask, anatomy)
            log_probs = torch.log(probs + 1e-8)
            loss = loss_type(log_probs, anatomy)
            total_loss += loss.item()
    
            pred = probs.argmax(dim=1)
            print(probs, anatomy)
            correct += (pred == anatomy).sum().item()
            total_num += anatomy.size(0)
            
    acc = correct / total_num
    print(f'[Validate Classifier] {correct:3d} out of {total_num:3d}, Acc: {acc:.4f}')

    return total_loss


        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # Data loader
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle = True, is_train = True)
    val_cls_loader = create_data_loaders(data_path = args.data_path_val, args = args, shuffle = True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, is_train = False)

    # Model and classifier init
    classifier = AnatomyClassifier(2, 2)
    classifier.to(device=device)

    model = ASPIN_VarNet(classifier, num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device=device)

    model_soft = SoftASPIN_VarNet(model, num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model_soft.to(device=device)

    ########################################
    ### Anatomy classifier training part ###

    # Maybe only classifiers???
    #optimizer_cls = torch.optim.Adam(model.parameters(), lr = args.lr)
    optimizer_cls = torch.optim.Adam(model.parameters(), lr = args.lr)
    loss_type_cls = nn.NLLLoss().to(device=device)

    for epoch in range(10):
        print(f'Classifier Epoch #{epoch:2d} ..............')
        
        train_loss = train_epoch_cls(args, epoch, model, train_loader, optimizer_cls, loss_type_cls)
        val_loss = validate_cls(args, model, val_cls_loader, loss_type_cls)

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        
        val_loss = val_loss / len(val_loader)

        print("Anatomomy Classifier training")
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g}',
        )

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Classifier training ended, now starting model training")
    
    ###########################
    ### Model training part ###
    
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0
    
    # bring checkpoint
    """
    checkpoint_path = '/root/result/augoff-epoch20to40/checkpoints/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Checkpoint loaded! Resume from epoch {start_epoch}")
    """
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        args.current_epoch = epoch
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        # Transition from ASPIN to Soft-ASPIN
        classifier_state_dict = model.anatomy_classifier.state_dict()
        for cascade in model_soft.cascades:
            cascade_classifier = cascade.model.aspin.classifier
            cascade_classifier.load_state_dict(classifier_state_dict)
        # transition end        
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model_soft, val_loader)
        
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

        save_model(args, args.exp_dir, epoch + 1, model_soft, optimizer, best_val_loss, is_new_best)
        print("Model training")
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
