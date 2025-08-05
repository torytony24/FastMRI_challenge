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
from utils.model.classifier import AnatomyClassifier


# Debugger: OFF 0 / ON 1
debugger = 0


import torchvision.utils as vutils

def save_output_image(tensor, filename, output_dir='/root/images'):
    
    os.makedirs(output_dir, exist_ok=True)

    if tensor.dim() == 4:  # [B, 1, H, W]
        img_tensor = tensor[0]  # [1, H, W]
    elif tensor.dim() == 3:  # [B, H, W]
        img_tensor = tensor[0].unsqueeze(0)  # [1, H, W]
    else:
        raise ValueError(f"Unsupported tensor shape {tensor.shape}")

    # Normalize to [0,1] for saving
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)

    # Save using torchvision utils
    vutils.save_image(img_tensor, os.path.join(output_dir, filename))

def crop_feature(tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    _, _, h, w = tensor.shape

    # --- Center crop if too big ---
    top = max((h - target_height) // 2, 0)
    left = max((w - target_width) // 2, 0)
    bottom = top + min(h, target_height)
    right = left + min(w, target_width)
    tensor = tensor[:, :, top:bottom, left:right]

    # --- Padding if too small ---
    pad_height = target_height - tensor.shape[2]
    pad_width = target_width - tensor.shape[3]
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return tensor



def train_epoch(args, model, classifier, data_loader, optimizer, loss_type):
    model.eval()
    classifier.train()
    len_loader = len(data_loader)
    total_loss = 0.

    cnt = 0
    for iter, data in enumerate(data_loader):
        cnt+=debugger
        if cnt>5:
            break
            
        mask, kspace, target, _, _, _, anatomy = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        anatomy = anatomy.cuda(non_blocking=True)

        output, feature = model(kspace, mask)
        feature = crop_feature(feature, 640, 368)
        #save_output_image(feature, f'output_iter_{iter}.png')
        probs = classifier(feature)
        loss = loss_type(probs, anatomy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = probs.argmax(dim=1)

        if iter % args.report_interval == 0:
            print( f'Train Classifier [{iter:4d}/{len(data_loader):4d}]' )
            printprobs = F.softmax(probs, dim=1)
            print( f'Prob: [{printprobs.squeeze().tolist()}], Pred: [{pred.squeeze().tolist()}], True: [{anatomy.squeeze().tolist()}], Loss: [{loss.item():.4f}]')
        


def validate(args, model, classifier, data_loader, loss_type):
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
    
            output, feature = model(kspace, mask)
            feature = crop_feature(feature, 640, 368)
            #save_output_image(output, f'output_iter_{iter}.png')
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



def save_model(args, epoch, model, best_val_loss, is_new_best):
    save_dir = '/root/FastMRI_challenge/Classifier_savefile/checkpoints/'
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'best_val_loss': best_val_loss,
        },
        f=save_dir + 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(save_dir+ 'model.pt', save_dir+ 'best_model.pt')




def train_classifier(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    classifier = AnatomyClassifier(2,2)
    classifier.to(device=device)

    model = VarNet(num_cascades=args.cascade, chans=args.chans, sens_chans=args.sens_chans)
    model.to(device=device)
    
    checkpoint_path = '/root/FastMRI_challenge/VarNet_savefile/checkpoints/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print("... VarNet for classifier loaded!")

    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle = True, is_train = True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, shuffle = True)

    optimizer = torch.optim.Adam(classifier.parameters(), lr = args.lr)
    loss_type = nn.CrossEntropyLoss().to(device=device)

    best_val_loss = 9999999.
    start_epoch = 0

    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print("@@@@@@@@@@@@@@@@@@@@@ [Train Anatomy Classifier] @@@@@@@@@@@@@@@@@@@@@")
        train_epoch(args, model, classifier, train_loader, optimizer, loss_type)

        print("@@@@@@@@@@@@@@@@@@@@@ [Validate Anatomy Classifier] @@@@@@@@@@@@@@@@@@@@@")
        val_loss = validate(args, model, classifier, val_loader, loss_type)
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        print(val_loss_log)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        save_model(args, epoch + 1, classifier, best_val_loss, is_new_best)





