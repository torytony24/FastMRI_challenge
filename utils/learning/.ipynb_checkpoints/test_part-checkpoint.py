import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.Student_varnet import Student_VarNet
from utils.learning.train_classifier import crop_feature

def test(args, model_cls, classifier, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices, _) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            output, feature = model_cls(kspace, mask)
            feature = crop_feature(feature, 640, 368)
            probs = classifier(feature)
            pred = probs.argmax(dim=1)
            output = model(kspace, mask, pred)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    model_cls = VarNet(num_cascades=args.cascade, chans=args.chans, sens_chans=args.sens_chans)
    model_cls.to(device=device)
    
    checkpoint_path = '/root/FastMRI_challenge/VarNet_savefile/checkpoints/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_cls.load_state_dict(checkpoint['model'])
    print("... VarNet for classifier loaded!")

    classifier = AnatomyClassifier(2,2)
    classifier.to(device=device)
    
    checkpoint_path = '/root/FastMRI_challenge/Classifier_savefile/checkpoints/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    classifier.load_state_dict(checkpoint['model'])
    print("... Classifier loaded!")

    model = Student_VarNet(num_cascades=args.cascade, chans=args.chans, sens_chans=args.sens_chans)
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model_cls, classifier, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)