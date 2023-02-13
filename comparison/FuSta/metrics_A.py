import sys

sys.path.append('core')

import os
import argparse
import sys
import torch
from PIL import Image
import numpy as np
import cv2
import math
import pdb
import matplotlib.pyplot as plt

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def load_image_list(image_files):
    images = []
    for imfile in image_files:
        images.append(load_image(imfile))

    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]

# RAFT
parser = argparse.ArgumentParser()
args = parser.parse_args()
flow_model = torch.nn.DataParallel(RAFT(args))
flow_model.load_state_dict(torch.load('raft_models/raft-things.pth'))
flow_model = flow_model.module
flow_model.to('cuda')
flow_model.eval()

def calc_flow(img1, img2):
    with torch.no_grad():

        images = load_image_list([img1, img2])

        flow_low, flow_up = flow_model(images[0, None], images[1, None], iters=20, test_mode=True)
    return flow_up


def metric_A(inn, out):
    in_src = sorted(glob.glob(inn + '*.png'))
    in_src = [x for x in in_src if 'mask.png' not in x]
    out_src = sorted(glob.glob(out + '*.png'))
    out_src = [x for x in out_src if 'mask.png' not in x]
    length = min(len(in_src), len(out_src))
    in_src = in_src[:length]
    out_src = out_src[:length]
    in_sum = 0.0
    out_sum = 0.0
    for i in range(length-1):
        print(in_src[i])
        print(in_src[i+1])
        in_flow = calc_flow(in_src[i], in_src[i+1])
        in_sum += (np.sqrt(np.maximum(np.sum(in_flow.cpu().numpy()), 1e-6)))
        print(out_src[i])
        print(out_src[i+1])
        out_flow = calc_flow(out_src[i], out_src[i+1])
        out_sum += (np.sqrt(np.maximum(np.sum(out_flow.cpu().numpy()), 1e-6)))
    a = out_sum / in_sum
    return a
