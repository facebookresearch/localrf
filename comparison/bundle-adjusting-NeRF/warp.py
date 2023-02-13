import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F

import util
from util import log,debug
import camera

def get_normalized_pixel_grid(opt):
    y_range = ((torch.arange(opt.H,dtype=torch.float32,device=opt.device)+0.5)/opt.H*2-1)*(opt.H/max(opt.H,opt.W))
    x_range = ((torch.arange(opt.W,dtype=torch.float32,device=opt.device)+0.5)/opt.W*2-1)*(opt.W/max(opt.H,opt.W))
    Y,X = torch.meshgrid(y_range,x_range) # [H,W]
    xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    xy_grid = xy_grid.repeat(opt.batch_size,1,1) # [B,HW,2]
    return xy_grid

def get_normalized_pixel_grid_crop(opt):
    y_crop = (opt.H//2-opt.H_crop//2,opt.H//2+opt.H_crop//2)
    x_crop = (opt.W//2-opt.W_crop//2,opt.W//2+opt.W_crop//2)
    y_range = ((torch.arange(*(y_crop),dtype=torch.float32,device=opt.device)+0.5)/opt.H*2-1)*(opt.H/max(opt.H,opt.W))
    x_range = ((torch.arange(*(x_crop),dtype=torch.float32,device=opt.device)+0.5)/opt.W*2-1)*(opt.W/max(opt.H,opt.W))
    Y,X = torch.meshgrid(y_range,x_range) # [H,W]
    xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    xy_grid = xy_grid.repeat(opt.batch_size,1,1) # [B,HW,2]
    return xy_grid

def warp_grid(opt,xy_grid,warp):
    if opt.warp.type=="translation":
        assert(opt.warp.dof==2)
        warped_grid = xy_grid+warp[...,None,:]
    elif opt.warp.type=="rotation":
        assert(opt.warp.dof==1)
        warp_matrix = lie.so2_to_SO2(warp)
        warped_grid = xy_grid@warp_matrix.transpose(-2,-1) # [B,HW,2]
    elif opt.warp.type=="rigid":
        assert(opt.warp.dof==3)
        xy_grid_hom = camera.to_hom(xy_grid)
        warp_matrix = lie.se2_to_SE2(warp)
        warped_grid = xy_grid_hom@warp_matrix.transpose(-2,-1) # [B,HW,2]
    elif opt.warp.type=="homography":
        assert(opt.warp.dof==8)
        xy_grid_hom = camera.to_hom(xy_grid)
        warp_matrix = lie.sl3_to_SL3(warp)
        warped_grid_hom = xy_grid_hom@warp_matrix.transpose(-2,-1)
        warped_grid = warped_grid_hom[...,:2]/(warped_grid_hom[...,2:]+1e-8) # [B,HW,2]
    else: assert(False)
    return warped_grid

def warp_corners(opt,warp_param):
    y_crop = (opt.H//2-opt.H_crop//2,opt.H//2+opt.H_crop//2)
    x_crop = (opt.W//2-opt.W_crop//2,opt.W//2+opt.W_crop//2)
    Y = [((y+0.5)/opt.H*2-1)*(opt.H/max(opt.H,opt.W)) for y in y_crop]
    X = [((x+0.5)/opt.W*2-1)*(opt.W/max(opt.H,opt.W)) for x in x_crop]
    corners = [(X[0],Y[0]),(X[0],Y[1]),(X[1],Y[1]),(X[1],Y[0])]
    corners = torch.tensor(corners,dtype=torch.float32,device=opt.device).repeat(opt.batch_size,1,1)
    corners_warped = warp_grid(opt,corners,warp_param)
    return corners_warped

def check_corners_in_range(opt,warp_param):
    corners_all = warp_corners(opt,warp_param)
    X = (corners_all[...,0]/opt.W*max(opt.H,opt.W)+1)/2*opt.W-0.5
    Y = (corners_all[...,1]/opt.H*max(opt.H,opt.W)+1)/2*opt.H-0.5
    return (0<=X).all() and (X<opt.W).all() and (0<=Y).all() and (Y<opt.H).all()

class Lie():

    def so2_to_SO2(self,theta): # [...,1]
        thetax = torch.stack([torch.cat([theta.cos(),-theta.sin()],dim=-1),
                              torch.cat([theta.sin(),theta.cos()],dim=-1)],dim=-2)
        R = thetax
        return R

    def SO2_to_so2(self,R): # [...,2,2]
        theta = torch.atan2(R[...,1,0],R[...,0,0])
        return theta[...,None]

    def so2_jacobian(self,X,theta): # [...,N,2],[...,1]
        dR_dtheta = torch.stack([torch.cat([-theta.sin(),-theta.cos()],dim=-1),
                                 torch.cat([theta.cos(),-theta.sin()],dim=-1)],dim=-2) # [...,2,2]
        J = X@dR_dtheta.transpose(-2,-1)
        return J[...,None] # [...,N,2,1]

    def se2_to_SE2(self,delta): # [...,3]
        u,theta = delta.split([2,1],dim=-1)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        V = torch.stack([torch.cat([A,-B],dim=-1),
                         torch.cat([B,A],dim=-1)],dim=-2)
        R = self.so2_to_SO2(theta)
        Rt = torch.cat([R,V@u[...,None]],dim=-1)
        return Rt

    def SE2_to_se2(self,Rt,eps=1e-7): # [...,2,3]
        R,t = Rt.split([2,1],dim=-1)
        theta = self.SO2_to_so2(R)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        denom = (A**2+B**2+eps)[...,None]
        invV = torch.stack([torch.cat([A,B],dim=-1),
                            torch.cat([-B,A],dim=-1)],dim=-2)/denom
        u = (invV@t)[...,0]
        delta = torch.cat([u,theta],dim=-1)
        return delta

    def se2_jacobian(self,X,delta): # [...,N,2],[...,3]
        u,theta = delta.split([2,1],dim=-1)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        D = self.taylor_D(theta)
        V = torch.stack([torch.cat([A,-B],dim=-1),
                         torch.cat([B,A],dim=-1)],dim=-2)
        R = self.so2_to_SO2(theta)
        dV_dtheta = torch.stack([torch.cat([C,-D],dim=-1),
                                 torch.cat([D,C],dim=-1)],dim=-2) # [...,2,2]
        dt_dtheta = dV_dtheta@u[...,None] # [...,2,1]
        J_so2 = self.so2_jacobian(X,theta) # [...,N,2,1]
        dX_dtheta = J_so2+dt_dtheta[...,None,:,:] # [...,N,2,1]
        dX_du = V[...,None,:,:].repeat(*[1]*(len(dX_dtheta.shape)-3),dX_dtheta.shape[-3],1,1)
        J = torch.cat([dX_du,dX_dtheta],dim=-1)
        return J # [...,N,2,3]

    def sl3_to_SL3(self,h):
        # homography: directly expand matrix exponential
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.6151&rep=rep1&type=pdf
        h1,h2,h3,h4,h5,h6,h7,h8 = h.chunk(8,dim=-1)
        A = torch.stack([torch.cat([h5,h3,h1],dim=-1),
                         torch.cat([h4,-h5-h6,h2],dim=-1),
                         torch.cat([h7,h8,h6],dim=-1)],dim=-2)
        H = A.matrix_exp()
        return H

    def taylor_A(self,x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_B(self,x,nth=10):
        # Taylor expansion of (1-cos(x))/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i+1)/denom
        return ans

    def taylor_C(self,x,nth=10):
        # Taylor expansion of (x*cos(x)-sin(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**(i+1)*x**(2*i+1)*(2*i+2)/denom
        return ans

    def taylor_D(self,x,nth=10):
        # Taylor expansion of (x*sin(x)+cos(x)-1)/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)*(2*i+1)/denom
        return ans

lie = Lie()
