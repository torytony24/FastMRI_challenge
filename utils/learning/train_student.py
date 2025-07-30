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
from utils.model.teacher_varnet import Teacher_VarNet
from utils.model.student_varnet import Student_VarNet
from utils.model.classifier import AnatomyClassifier


# Debugger: OFF 0 / ON 1
debugger = 0


def train_epoch(args, epoch, model_teacher_brain, model_teacher_knee, model_student, data_loader, optimizer, loss_recon, loss_distill):
    model_student.train()
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

        with torch.no_grad():
            if anatomy == 0:
                _, feature_teacher = model_teacher_brain(kspace, mask)
            elif anatomy == 1:
                _, feature_teacher = model_teacher_knee(kspace, mask)
            
        output, feature_student = model_student(kspace, mask, anatomy)

        loss_1 = loss_recon(output, target, maximum)
        loss_2 = loss_distill(feature_student, feature_teacher)

        alpha = 0.1
        loss = loss_1 + alpha * loss_2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'L1 = [{loss_1.item():.4g}] / L2 = [{loss_2.item():.4g}]',
            )
            
    total_loss = total_loss / len_loader
    return total_loss

def validate(args, model_student, model_cls, classifier, data_loader):
    model_student.eval()
    model_cls.eval()
    classifier.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)

    with torch.no_grad():
        cnt=0
        for iter, data in enumerate(data_loader):
            if cnt>5:
                break
            cnt+=debugger
            
            mask, kspace, target, _, fnames, slices, anatomy = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            anatomy = anatomy.cuda(non_blocking=True)

            _, feature = model_cls(kspace, mask)
            probs = classifier(feature)
            pred = probs.argmax(dim=1)
            #print( f'Pred: [{pred.squeeze().tolist()}], True: [{anatomy.squeeze().tolist()}]')
            #output, _ = model_student(kspace, mask, pred)
            output, _ = model_student(kspace, mask, anatomy)

            
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


def train_student(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    ##############################
    ######### Classifier #########
    classifier = AnatomyClassifier(2, 2)
    classifier.to(device=device)

    model_cls = VarNet(num_cascades=args.cascade, 
                       chans=args.chans, sens_chans=args.sens_chans)
    model_cls.to(device=device)
    
    classifier_path = '/root/FastMRI_challenge/checkpoints_cls/best_model_cls.pt'
    checkpoint_cls = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(checkpoint_cls['classifier'])
    print("... Classifier loaded!")

    
    ##################################
    ####### Load teacher model #######

    model_teacher_brain = Teacher_VarNet(num_cascades=args.cascade, 
                                         chans=args.chans, sens_chans=args.sens_chans)
    model_teacher_knee = Teacher_VarNet(num_cascades=args.cascade, 
                                         chans=args.chans, sens_chans=args.sens_chans)
    model_teacher_brain.to(device=device)
    model_teacher_knee.to(device=device)

    """
    def inspect_teacher_model(path):
        checkpoint = torch.load(path, map_location='cpu')
        keys = list(checkpoint['model'].keys())
        print(f"Total keys: {len(keys)}")
        cascade_keys = [k for k in keys if k.startswith("cascades.")]
        cascade_ids = sorted(set(k.split('.')[1] for k in cascade_keys))
        print(f"Cascade modules present: {cascade_ids}")
    inspect_teacher_model('/root/result/teacher-brain-savefile/checkpoints/best_model.pt')
    """
    
    teacher_brain_path = '/root/result/teacher-brain-savefile/checkpoints/best_model.pt'
    teacher_knee_path = '/root/result/teacher-knee-savefile/checkpoints/best_model.pt'

    teacher_brain_checkpoint = torch.load(teacher_brain_path, map_location=device)
    teacher_knee_checkpoint = torch.load(teacher_knee_path, map_location=device)

    model_teacher_brain.load_state_dict(teacher_brain_checkpoint['model'])
    model_teacher_knee.load_state_dict(teacher_knee_checkpoint['model'])

    model_teacher_brain.eval()
    model_teacher_knee.eval()
    
    print("... Teacher model loaded!")

    
    ###########################
    ### Model training part ###
    model_student = Student_VarNet(num_cascades=args.cascade, 
                                   chans=args.chans, sens_chans=args.sens_chans)
    model_student.to(device=device)
    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle = True, is_train = True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, shuffle = True, is_train = False)

    optimizer = torch.optim.Adam(model_student.parameters(), args.lr)
    loss_recon = SSIMLoss().to(device=device)
    loss_distill = nn.MSELoss().to(device=device)

    best_val_loss = 1.
    start_epoch = 0
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        args.current_epoch = epoch

        print("@@@@@@@@@@@@@@@@@@@@@ [Train Model] @@@@@@@@@@@@@@@@@@@@@")
        train_loss = train_epoch(args, epoch, model_teacher_brain, model_teacher_knee, model_student, train_loader, optimizer, loss_recon, loss_distill)
        
        print("@@@@@@@@@@@@@@@@@@@@@ [Validate Model] @@@@@@@@@@@@@@@@@@@@@")
        val_loss, num_subjects, reconstructions, targets, inputs = validate(args, model_student, model_cls, classifier, val_loader)
        
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

        save_model(args, args.exp_dir, epoch + 1, model_student, optimizer, best_val_loss, is_new_best)
        print( f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} ValLoss = {val_loss:.4g}' , )

        print(val_loss_log)
        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@ New Record @@@@@@@@@@@@@@@@@@@@@")
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)

