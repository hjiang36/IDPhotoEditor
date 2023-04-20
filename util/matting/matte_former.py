import numpy as np
import torch
from torch.nn import functional as F

import sys
import os
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).parent.absolute(), "matteformer"))

from .matteformer import networks as matteformer_networks
from .matteformer import utils as matteformer_utils


"""
Prepare the image and trimap for the matte former model.

@param image: the image to be matted.
@param trimap: the trimap of the image (0 for background, 1 for foreground, 0.5 for unknown).

@return: the image and trimap in the correct format for the matte former model.
"""
def generator_tensor_dict(image: np.ndarray, trimap: np.ndarray) -> dict:
    trimap = (trimap * 255.0).astype(np.uint8)
    sample = {'image': image, 'trimap':trimap, 'alpha_shape':(image.shape[0], image.shape[1])}

    # reshape
    h, w = sample["alpha_shape"]
    
    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,32), (32, 32)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
    
    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image, trimap = sample['image'], sample['trimap']

    # swap color axis
    image = image.transpose((2, 0, 1))

    # trimap configuration
    padded_trimap[padded_trimap < 85] = 0
    padded_trimap[padded_trimap >= 170] = 2
    padded_trimap[padded_trimap >= 85] = 1

    # to tensor
    sample['image'], sample['trimap'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long)
    sample['image'] = sample['image'].sub_(mean).div_(std)

    # trimap to one-hot 3 channel
    sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()

    # add first channel
    sample['image'], sample['trimap'] = sample['image'][None, ...], sample['trimap'][None, ...]

    return sample


class MatteFormerMatting:
    """
    Create class for matte former matting algorithm on provided image and trimap.

    @param model_path: the path to the matte former model trained weights.
    """
    def __init__(self, model_path: str):
        # build model
        self.model = matteformer_networks.get_generator(is_train=False)
        self.model.cuda()

        # load checkpoint
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(
            matteformer_utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
        self.model.eval()

    
    """
    Run matte former matting algorithm on provided image and trimap.

    @param img: the image to be matted.
    @param trimap: the trimap of the image (0 for background, 1 for foreground, 0.5 for unknown).

    @return: the matted alpha mask.
    """
    def run_matte_former_matting(self, img, trimap) -> np.ndarray:
        image_dict = generator_tensor_dict(img, trimap)
        with torch.no_grad(): 
            image, trimap = image_dict['image'], image_dict['trimap']
            image = image.cuda()
            trimap = trimap.cuda()

            # run model
            pred = self.model(image, trimap)
            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

            # refinement
            alpha_pred = alpha_pred_os8.clone().detach()
            weight_os4 = matteformer_utils.get_unknown_tensor_from_pred(
                alpha_pred, rand_width=matteformer_utils.CONFIG.model.self_refine_width1,
                train_mode=False)
            alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
            weight_os1 = matteformer_utils.get_unknown_tensor_from_pred(
                alpha_pred, rand_width=matteformer_utils.CONFIG.model.self_refine_width2,
                train_mode=False)
            alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]

            h, w = image_dict['alpha_shape']
            alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
            alpha_pred = alpha_pred.astype(np.uint8)

            alpha_pred[np.argmax(trimap.cpu().numpy()[0], axis=0) == 0] = 0
            alpha_pred[np.argmax(trimap.cpu().numpy()[0], axis=0) == 2] = 255

            alpha_pred = alpha_pred[32:h+32, 32:w+32]
        return alpha_pred.astype(np.float32) / 255.0