import numpy as np
import os,sys,time
import torch
import importlib

import options
from util import log

def main():

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for evaluating NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)

    with torch.cuda.device(opt.device):

        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)

        m.load_dataset(opt,eval_split="test")
        m.build_networks(opt)

        # if opt.model=="barf":
        #     m.generate_videos_pose(opt)

        m.restore_checkpoint(opt)
        # if opt.data.dataset in ["blender","llff"]:
        m.evaluate_full(opt)
        # m.generate_videos_synthesis(opt)

if __name__=="__main__":
    main()
