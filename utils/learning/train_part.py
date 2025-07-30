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
from utils.model.varnet import VarNet
#from utils.model.aspin_varnet import ASPIN_VarNet
from utils.model.teacher_varnet import Teacher_VarNet
from utils.model.classifier import AnatomyClassifier


# Debugger: OFF 0 / ON 1
debugger = 1


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, train_identifier):
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
        
        #output = model(kspace, mask, anatomy)
        output, _ = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} ',
            )
            
    total_loss = total_loss / len_loader
    return total_loss

def validate(args, model, model_cls, classifier, data_loader, train_identifier):
    model.eval()
    model_cls.eval()
    classifier.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)

    with torch.no_grad():
        cnt=0
        for iter, data in enumerate(data_loader):
            if cnt>40:
                break
            cnt+=debugger
            mask, kspace, target, _, fnames, slices, anatomy = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            anatomy = anatomy.cuda(non_blocking=True)

            if train_identifier != anatomy:
                continue
            
            _, feature = model_cls(kspace, mask)
            probs = classifier(feature)
            pred = probs.argmax(dim=1)
            print( f'Pred: [{pred.squeeze().tolist()}], True: [{anatomy.squeeze().tolist()}]')
            #output = model(kspace, mask, pred)
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

"""
def train_epoch_cls(args, model, classifier, data_loader, optimizer, loss_type):
    model.eval()
    classifier.train()
    len_loader = len(data_loader)
    total_loss = 0.

    cnt = 0
    for iter, data in enumerate(data_loader):
        cnt+=debugger
        if cnt>5:
            break
        mask, kspace, _, _, _, _, anatomy = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        anatomy = anatomy.cuda(non_blocking=True)

        _, feature = model(kspace, mask)
        
        probs = classifier(feature)
        loss = loss_type(probs, anatomy)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = probs.argmax(dim=1)

        if iter % args.report_interval == 0:
            print( f'Train Classifier [{iter:4d}/{len(data_loader):4d}]' )
            printprobs = F.softmax(probs, dim=1)
            print( f'Prob: {printprobs.squeeze().tolist()}, Pred: [{pred.squeeze().tolist()}], True: [{anatomy.squeeze().tolist()}]')
        


def validate_cls(args, model, classifier, data_loader, loss_type):
    model.eval()
    classifier.eval()
    correct = 0
    total_num = 0
    total_loss = 0

    with torch.no_grad():
        cnt=0
        for iter, data in enumerate(data_loader):
            if cnt>5:
                break
            cnt+=debugger
            mask, kspace, _, _, _, _, anatomy = data
            mask = mask.cuda(non_blocking=True)
            kspace = kspace.cuda(non_blocking=True)
            anatomy = anatomy.cuda(non_blocking=True)
    
            _, feature = model(kspace, mask)
            
            probs = classifier(feature)

            loss = loss_type(probs, anatomy)
    
            pred = probs.argmax(dim=1)
            correct += (pred == anatomy).sum().item()
            total_num += anatomy.size(0)
            total_loss += loss.item()

            if iter % args.report_interval == 0:
                print( f'Validate Classifier [{iter:4d}/{len(data_loader):4d}]' )
                printprobs = F.softmax(probs, dim=1)
                print( f'Prob: {printprobs.squeeze().tolist()}, Pred: [{pred.squeeze().tolist()}], True: [{anatomy.squeeze().tolist()}]')
                acc = correct / total_num
                print(f'Rate: [{correct:3d} out of {total_num:3d}], Acc: [{acc*100:.2f}%]')

    return total_loss

def save_model_cls(args, epoch, model, best_val_loss, is_new_best):
    save_dir = '/root/FastMRI_challenge/checkpoints_cls/'
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'classifier': model.state_dict(),
            'best_cls_loss': best_val_loss,
        },
        f=save_dir + 'model_cls.pt'
    )
    if is_new_best:
        shutil.copyfile(save_dir + 'model_cls.pt', save_dir + 'best_model_cls.pt')
"""


def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # Model and classifier init
    classifier = AnatomyClassifier(2, 2)
    classifier.to(device=device)

    model_cls = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model_cls.to(device=device)

    """
    # Data loader
    train_loader_cls = create_data_loaders(data_path = args.data_path_train, args = args, shuffle = True, is_train = True)
    val_loader_cls = create_data_loaders(data_path = args.data_path_val, args = args, shuffle = True)

    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr = args.lr)
    loss_type_cls = nn.CrossEntropyLoss().to(device=device)

    ########################################
    ### Anatomy classifier training part ###
    
    best_cls_loss = 9999999.
    cls_loss_log = np.empty((0, 2))
    for epoch in range(9):
        print("@@@@@@@@@@@@@@@@@@@@@ [Train Anatomy Classifier] @@@@@@@@@@@@@@@@@@@@@")
        train_epoch_cls(args, model_cls, classifier, train_loader_cls, optimizer_cls, loss_type_cls)
    
        print("@@@@@@@@@@@@@@@@@@@@@ [Validate Anatomy Classifier] @@@@@@@@@@@@@@@@@@@@@")
        cls_loss = validate_cls(args, model_cls, classifier, val_loader_cls, loss_type_cls)
        cls_loss_log = np.append(cls_loss_log, np.array([[epoch, cls_loss]]), axis=0)
        print(cls_loss_log)

        is_new_best = cls_loss < best_cls_loss
        best_cls_loss = min(best_cls_loss, cls_loss)
        save_model_cls(args, epoch + 1, classifier, best_cls_loss, is_new_best)
    """

    
    #############################
    ### Bring best classifier ###
    
    classifier_path = '/root/FastMRI_challenge/checkpoints_cls/best_model_cls.pt'
    checkpoint_cls = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(checkpoint_cls['classifier'])
    print(f"@@@@@@@@@@@@@@@@ Classifier loaded! / epoch: {checkpoint_cls['epoch']:3d} / Loss: {checkpoint_cls['best_cls_loss']:.4f} @@@@@@@@@@@@@@@@")

    
    ###########################
    ### Model training part ###
    model = Teacher_VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device=device)
    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle = True, is_train = True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, is_train = False)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    loss_type = SSIMLoss().to(device=device)

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
    
    # teacher: brain 0 / knee 1
    train_identifier = 0
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        args.current_epoch = epoch

        print("@@@@@@@@@@@@@@@@@@@@@ [Train Model] @@@@@@@@@@@@@@@@@@@@@")
        train_loss = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, train_identifier)
        
        print("@@@@@@@@@@@@@@@@@@@@@ [Validate Model] @@@@@@@@@@@@@@@@@@@@@")
        val_loss, num_subjects, reconstructions, targets, inputs = validate(args, model, model_cls, classifier, val_loader, train_identifier)
        
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

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print( f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} ValLoss = {val_loss:.4g}' , )

        print(val_loss_log)
        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@ New Record @@@@@@@@@@@@@@@@@@@@@")
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)

