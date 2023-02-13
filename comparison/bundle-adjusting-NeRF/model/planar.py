import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import PIL
import PIL.Image,PIL.ImageDraw
import imageio

import util,util_vis
from util import log,debug
from . import base
import warp

# ============================ main engine for training and evaluation ============================

class Model(base.Model):

    def __init__(self,opt):
        super().__init__(opt)
        opt.H_crop,opt.W_crop = opt.data.patch_crop

    def load_dataset(self,opt,eval_split=None):
        image_raw = PIL.Image.open(opt.data.image_fname)
        self.image_raw = torchvision_F.to_tensor(image_raw).to(opt.device)

    def build_networks(self,opt):
        super().build_networks(opt)
        self.graph.warp_param = torch.nn.Embedding(opt.batch_size,opt.warp.dof).to(opt.device)
        torch.nn.init.zeros_(self.graph.warp_param.weight)

    def setup_optimizer(self,opt):
        log.info("setting up optimizers...")
        optim_list = [
            dict(params=self.graph.neural_image.parameters(),lr=opt.optim.lr),
            dict(params=self.graph.warp_param.parameters(),lr=opt.optim.lr_warp),
        ]
        optimizer = getattr(torch.optim,opt.optim.algo)
        self.optim = optimizer(optim_list)
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched.type)
            kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim,**kwargs)

    def setup_visualizer(self,opt):
        super().setup_visualizer(opt)
        # set colors for visualization
        box_colors = ["#ff0000","#40afff","#9314ff","#ffd700","#00ff00"]
        box_colors = list(map(util.colorcode_to_number,box_colors))
        self.box_colors = np.array(box_colors).astype(int)
        assert(len(self.box_colors)==opt.batch_size)
        # create visualization directory
        self.vis_path = "{}/vis".format(opt.output_path)
        os.makedirs(self.vis_path,exist_ok=True)
        self.video_fname = "{}/vis.mp4".format(opt.output_path)

    def train(self,opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.ep = self.it = self.vis_it = 0
        self.graph.train()
        var = edict(idx=torch.arange(opt.batch_size))
        # pre-generate perturbations
        self.warp_pert,var.image_pert = self.generate_warp_perturbation(opt)
        # train
        var = util.move_to_device(var,opt.device)
        loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
        # visualize initial state
        var = self.graph.forward(opt,var)
        self.visualize(opt,var,step=0)
        for it in loader:
            # train iteration
            loss = self.train_iteration(opt,var,loader)
            if opt.warp.fix_first:
                self.graph.warp_param.weight.data[0] = 0
        # after training
        os.system("ffmpeg -y -framerate 30 -i {}/%d.png -pix_fmt yuv420p {}".format(self.vis_path,self.video_fname))
        self.save_checkpoint(opt,ep=None,it=self.it)
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

    def train_iteration(self,opt,var,loader):
        loss = super().train_iteration(opt,var,loader)
        self.graph.neural_image.progress.data.fill_(self.it/opt.max_iter)
        return loss

    def generate_warp_perturbation(self,opt):
        # pre-generate perturbations (translational noise + homography noise)
        warp_pert_all = torch.zeros(opt.batch_size,opt.warp.dof,device=opt.device)
        trans_pert = [(0,0)]+[(x,y) for x in (-opt.warp.noise_t,opt.warp.noise_t)
                                    for y in (-opt.warp.noise_t,opt.warp.noise_t)]
        def create_random_perturbation():
            warp_pert = torch.randn(opt.warp.dof,device=opt.device)*opt.warp.noise_h
            warp_pert[0] += trans_pert[i][0]
            warp_pert[1] += trans_pert[i][1]
            return warp_pert
        for i in range(opt.batch_size):
            warp_pert = create_random_perturbation()
            while not warp.check_corners_in_range(opt,warp_pert[None]):
                warp_pert = create_random_perturbation()
            warp_pert_all[i] = warp_pert
        if opt.warp.fix_first:
            warp_pert_all[0] = 0
        # create warped image patches
        xy_grid = warp.get_normalized_pixel_grid_crop(opt) # [B,HW,2]
        xy_grid_warped = warp.warp_grid(opt,xy_grid,warp_pert_all)
        xy_grid_warped = xy_grid_warped.view([opt.batch_size,opt.H_crop,opt.W_crop,2])
        xy_grid_warped = torch.stack([xy_grid_warped[...,0]*max(opt.H,opt.W)/opt.W,
                                      xy_grid_warped[...,1]*max(opt.H,opt.W)/opt.H],dim=-1)
        image_raw_batch = self.image_raw.repeat(opt.batch_size,1,1,1)
        image_pert_all = torch_F.grid_sample(image_raw_batch,xy_grid_warped,align_corners=False)
        return warp_pert_all,image_pert_all

    def visualize_patches(self,opt,warp_param):
        image_pil = torchvision_F.to_pil_image(self.image_raw).convert("RGBA")
        draw_pil = PIL.Image.new("RGBA",image_pil.size,(0,0,0,0))
        draw = PIL.ImageDraw.Draw(draw_pil)
        corners_all = warp.warp_corners(opt,warp_param)
        corners_all[...,0] = (corners_all[...,0]/opt.W*max(opt.H,opt.W)+1)/2*opt.W-0.5
        corners_all[...,1] = (corners_all[...,1]/opt.H*max(opt.H,opt.W)+1)/2*opt.H-0.5
        for i,corners in enumerate(corners_all):
            P = [tuple(float(n) for n in corners[j]) for j in range(4)]
            draw.line([P[0],P[1],P[2],P[3],P[0]],fill=tuple(self.box_colors[i]),width=3)
        image_pil.alpha_composite(draw_pil)
        image_tensor = torchvision_F.to_tensor(image_pil.convert("RGB"))
        return image_tensor

    @torch.no_grad()
    def predict_entire_image(self,opt):
        xy_grid = warp.get_normalized_pixel_grid(opt)[:1]
        rgb = self.graph.neural_image.forward(opt,xy_grid) # [B,HW,3]
        image = rgb.view(opt.H,opt.W,3).detach().cpu().permute(2,0,1)
        return image

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        # compute PSNR
        psnr = -10*loss.render.log10()
        self.tb.add_scalar("{0}/{1}".format(split,"PSNR"),psnr,step)
        # warp error
        warp_error = (self.graph.warp_param.weight-self.warp_pert).norm(dim=-1).mean()
        self.tb.add_scalar("{0}/{1}".format(split,"warp error"),warp_error,step)

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train"):
        # dump frames for writing to video
        frame_GT = self.visualize_patches(opt,self.warp_pert)
        frame = self.visualize_patches(opt,self.graph.warp_param.weight)
        frame2 = self.predict_entire_image(opt)
        frame_cat = (torch.cat([frame,frame2],dim=1)*255).byte().permute(1,2,0).numpy()
        imageio.imsave("{}/{}.png".format(self.vis_path,self.vis_it),frame_cat)
        self.vis_it += 1
        # visualize in Tensorboard
        if opt.tb:
            colors = self.box_colors
            util_vis.tb_image(opt,self.tb,step,split,"image_pert",util_vis.color_border(var.image_pert,colors))
            util_vis.tb_image(opt,self.tb,step,split,"rgb_warped",util_vis.color_border(var.rgb_warped_map,colors))
            util_vis.tb_image(opt,self.tb,self.it+1,"train","image_boxes",frame[None])
            util_vis.tb_image(opt,self.tb,self.it+1,"train","image_boxes_GT",frame_GT[None])
            util_vis.tb_image(opt,self.tb,self.it+1,"train","image_entire",frame2[None])

# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.neural_image = NeuralImageFunction(opt)

    def forward(self,opt,var,mode=None):
        xy_grid = warp.get_normalized_pixel_grid_crop(opt)
        xy_grid_warped = warp.warp_grid(opt,xy_grid,self.warp_param.weight)
        # render images
        var.rgb_warped = self.neural_image.forward(opt,xy_grid_warped) # [B,HW,3]
        var.rgb_warped_map = var.rgb_warped.view(opt.batch_size,opt.H_crop,opt.W_crop,3).permute(0,3,1,2) # [B,3,H,W]
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        if opt.loss_weight.render is not None:
            image_pert = var.image_pert.view(opt.batch_size,3,opt.H_crop*opt.W_crop).permute(0,2,1)
            loss.render = self.MSE_loss(var.rgb_warped,image_pert)
        return loss

class NeuralImageFunction(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed

    def define_network(self,opt):
        input_2D_dim = 2+4*opt.arch.posenc.L_2D if opt.arch.posenc else 2
        # point-wise RGB prediction
        self.mlp = torch.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_2D_dim
            if li in opt.arch.skip: k_in += input_2D_dim
            linear = torch.nn.Linear(k_in,k_out)
            if opt.barf_c2f and li==0:
                # rescale first layer init (distribution was for pos.enc. but only xy is first used)
                scale = np.sqrt(input_2D_dim/2.)
                linear.weight.data *= scale
                linear.bias.data *= scale
            self.mlp.append(linear)

    def forward(self,opt,coord_2D): # [B,...,3]
        if opt.arch.posenc:
            points_enc = self.positional_encoding(opt,coord_2D,L=opt.arch.posenc.L_2D)
            points_enc = torch.cat([coord_2D,points_enc],dim=-1) # [B,...,6L+3]
        else: points_enc = coord_2D
        feat = points_enc
        # extract implicit features
        for li,layer in enumerate(self.mlp):
            if li in opt.arch.skip: feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li!=len(self.mlp)-1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_() # [B,...,3]
        return rgb

    def positional_encoding(self,opt,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=opt.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if opt.barf_c2f is not None:
            # set weights for different frequency bands
            start,end = opt.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*L
            k = torch.arange(L,dtype=torch.float32,device=opt.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,L)*weight).view(*shape)
        return input_enc
