import numpy as np
import torch


#debug
import matplotlib.pyplot as plt
import os
import torch
from utils.model.fastmri.fftc import ifft2c_new as ifft2c

def complex_abs(tensor):
    return torch.sqrt(tensor[..., 0] ** 2 + tensor[..., 1] ** 2)

def kspace_to_image(kspace):
    image = ifft2c(kspace)
    return complex_abs(image)

def visualize_kspace_and_image(kspace, target=None, title="", save_dir="/root/vis_debug", prefix=""):
    if kspace.dim() == 4:
        kspace = kspace[0]
    if target is not None and target.dim() == 3:
        target = target[0]

    image = kspace_to_image(kspace)
    kspace_log = torch.log1p(complex_abs(kspace))

    os.makedirs(save_dir, exist_ok=True)

    def save_image(img_tensor, filename):
        path = os.path.join(save_dir, f"{prefix}_{filename}.png")
        plt.imsave(path, img_tensor.cpu().numpy(), cmap='gray')
        print(f"Saved: {path}")
    save_image(kspace_log, f"{title}_kspace")
    save_image(image, f"{title}_image")
    if target is not None:
        save_image(target, f"{title}_target")
#debug end



def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, input, target, attrs, fname, slice, anatomy):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        # Add anatomy index
        if anatomy == "brain":
            anatomy_idx = 0
        elif anatomy == "knee":
            anatomy_idx = 1
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace, target, maximum, fname, slice, anatomy_idx    # Add anatomy index

# New class for augmentation

class AugDataTransform:
    def __init__(self, isforward, max_key, augmentor=None):
        self.isforward = isforward
        self.max_key = max_key
        self.augmentor = augmentor

    def __call__(self, mask, input, target, attrs, fname, slice, anatomy):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        # Add anatomy index
        if anatomy == "brain":
            anatomy_idx = 0
        elif anatomy == "knee":
            anatomy_idx = 1
        
        kspace = to_tensor(input)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)

        """
        # visualize before aug
        if self.augmentor is not None and not self.isforward:
            visualize_kspace_and_image(
                kspace=kspace,
                target=target,
                title="before_aug",
                prefix=f"{fname}_{slice}"
            )
        """
        
        
        # Add augmentation
        if self.augmentor is not None:
            target_size = [384,384]
            kspace, aug_target = self.augmentor(kspace, target_size)
            if aug_target is not None:
                target = aug_target

        """
        # visualize after aug
        if self.augmentor is not None and not self.isforward:
            visualize_kspace_and_image(
                kspace=kspace,
                target=target,
                title="after_aug",
                prefix=f"{fname}_{slice}"
            )
        """


        mask_tensor = torch.tensor(mask).to(kspace.device).unsqueeze(-1)
        kspace = kspace * mask_tensor
        
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()

        
        return mask, kspace, target, maximum, fname, slice, anatomy_idx

