import sys
import os
import requests

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('..')
import models_mae

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])



def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.0)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    out = torch.clip((x[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    out = out.numpy()
    out = cv2.cvtColor(np.uint8(out), cv2.COLOR_BGR2RGB)
    cv2.imwrite('./original.png',out)

    out = torch.clip((im_masked[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    out = out.numpy()
    out = cv2.cvtColor(np.uint8(out), cv2.COLOR_BGR2RGB)
    cv2.imwrite('./masked.png',out)

    out = torch.clip((y[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    out = out.numpy()
    out = cv2.cvtColor(np.uint8(out), cv2.COLOR_BGR2RGB)
    cv2.imwrite('./reconstruction.png',out)

    out = torch.clip((im_paste[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    out = out.numpy()
    out = cv2.cvtColor(np.uint8(out), cv2.COLOR_BGR2RGB)
    cv2.imwrite('./reconstruction+visible.png',out)





# load an image
img = Image.open('/mnt/ljk/liuwei/basic_codes/test_input_folder/xi.jpg')
img = img.resize((224, 224))
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std


# This is an MAE model trained with pixels as targets for visualization (ViT-Large, training mask ratio=0.75)

# download checkpoint if not exist

# chkpt_dir = 'mae_visualize_vit_large.pth'
# model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
# print('Model loaded.')

# # make random mask reproducible (comment out to make it change)
# torch.manual_seed(2)
# print('MAE with pixel reconstruction:')
# run_one_image(img, model_mae)

# This is an MAE model trained with an extra GAN loss for more realistic generation (ViT-Large, training mask ratio=0.75)

# download checkpoint if not exist

chkpt_dir = 'mae_visualize_vit_large_ganloss.pth'
model_mae_gan = prepare_model('mae_visualize_vit_large_ganloss.pth', 'mae_vit_large_patch16')
print('Model loaded.')


# make random mask reproducible (comment out to make it change)
torch.manual_seed(2)
print('MAE with extra GAN loss:')
run_one_image(img, model_mae_gan)


