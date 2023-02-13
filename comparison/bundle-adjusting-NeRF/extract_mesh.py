"""Extracts a 3D mesh from a pretrained model using marching cubes."""

import importlib
import sys

import numpy as np
import options
import torch
import tqdm
import trimesh
import mcubes

from util import log,debug

opt_cmd = options.parse_arguments(sys.argv[1:])
opt = options.set(opt_cmd=opt_cmd)

with torch.cuda.device(opt.device),torch.no_grad():

    model = importlib.import_module("model.{}".format(opt.model))
    m = model.Model(opt)

    m.build_networks(opt)
    m.restore_checkpoint(opt)

    t = torch.linspace(*opt.trimesh.range,opt.trimesh.res+1) # the best range might vary from model to model
    query = torch.stack(torch.meshgrid(t,t,t),dim=-1)
    query_flat = query.view(-1,3)

    density_all = []
    for i in tqdm.trange(0,len(query_flat),opt.trimesh.chunk_size,leave=False):
        points = query_flat[None,i:i+opt.trimesh.chunk_size].to(opt.device)
        ray_unit = torch.zeros_like(points) # dummy ray to comply with interface, not used
        _,density_samples = m.graph.nerf.forward(opt,points,ray_unit=ray_unit,mode=None)
        density_all.append(density_samples.cpu())
    density_all = torch.cat(density_all,dim=1)[0]
    density_all = density_all.view(*query.shape[:-1]).numpy()

    log.info("running marching cubes...")
    vertices,triangles = mcubes.marching_cubes(density_all,opt.trimesh.thres)
    vertices_centered = vertices/opt.trimesh.res-0.5
    mesh = trimesh.Trimesh(vertices_centered,triangles)

    obj_fname = "{}/mesh.obj".format(opt.output_path)
    log.info("saving 3D mesh to {}...".format(obj_fname))
    mesh.export(obj_fname)
