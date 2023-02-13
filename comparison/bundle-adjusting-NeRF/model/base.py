import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import torch.utils.tensorboard
import visdom
import importlib
import tqdm
from easydict import EasyDict as edict

import util,util_vis
from util import log,debug

# ============================ main engine for training and evaluation ============================

class Model():

    def __init__(self,opt):
        super().__init__()
        os.makedirs(opt.output_path,exist_ok=True)

    def load_dataset(self,opt,eval_split="val"):
        data = importlib.import_module("data.{}".format(opt.data.dataset))
        log.info("loading training data...")
        self.train_data = data.Dataset(opt,split="train",subset=opt.data.train_sub)
        self.train_loader = self.train_data.setup_loader(opt,shuffle=True)
        log.info("loading test data...")
        eval_split = "test"
        self.test_data = data.Dataset(opt,split=eval_split,subset=opt.data.val_sub)
        self.test_loader = self.test_data.setup_loader(opt,shuffle=False)

    def build_networks(self,opt):
        graph = importlib.import_module("model.{}".format(opt.model))
        log.info("building networks...")
        self.graph = graph.Graph(opt).to(opt.device)

    def setup_optimizer(self,opt):
        log.info("setting up optimizers...")
        optimizer = getattr(torch.optim,opt.optim.algo)
        self.optim = optimizer([dict(params=self.graph.parameters(),lr=opt.optim.lr)])
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched.type)
            kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim,**kwargs)

    def restore_checkpoint(self,opt):
        epoch_start,iter_start = None,None
        if opt.resume:
            log.info("resuming from previous checkpoint...")
            epoch_start,iter_start = util.restore_checkpoint(opt,self,resume=opt.resume)
        elif opt.load is not None:
            log.info("loading weights from checkpoint {}...".format(opt.load))
            epoch_start,iter_start = util.restore_checkpoint(opt,self,load_name=opt.load)
        else:
            log.info("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    def setup_visualizer(self,opt):
        log.info("setting up visualizers...")
        if opt.tb:
            self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=opt.output_path,flush_secs=10)
        if opt.visdom:
            # check if visdom server is runninng
            is_open = util.check_socket_open(opt.visdom.server,opt.visdom.port)
            retry = None
            while not is_open:
                retry = "n" #input("visdom port ({}) not open, retry? (y/n) ".format(opt.visdom.port))
                if retry not in ["y","n"]: continue
                if retry=="y":
                    is_open = util.check_socket_open(opt.visdom.server,opt.visdom.port)
                else: break
            # self.vis = visdom.Visdom(server=opt.visdom.server,port=opt.visdom.port,env=opt.group)

    def train(self,opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.it = self.iter_start
        # training
        if self.iter_start==0: self.validate(opt,ep=0)
        for self.ep in range(self.epoch_start,opt.max_epoch):
            self.train_epoch(opt)
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        # if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

    def train_epoch(self,opt):
        # before train epoch
        self.graph.train()
        # train epoch
        loader = tqdm.tqdm(self.train_loader,desc="training epoch {}".format(self.ep+1),leave=False)
        for batch in loader:
            # train iteration
            var = edict(batch)
            var = util.move_to_device(var,opt.device)
            loss = self.train_iteration(opt,var,loader)
        # after train epoch
        lr = self.sched.get_last_lr()[0] if opt.optim.sched else opt.optim.lr
        log.loss_train(opt,self.ep+1,lr,loss.all,self.timer)
        if opt.optim.sched: self.sched.step()
        if (self.ep+1)%opt.freq.val==0: self.validate(opt,ep=self.ep+1)
        if (self.ep+1)%opt.freq.ckpt==0: self.save_checkpoint(opt,ep=self.ep+1,it=self.it)

    def train_iteration(self,opt,var,loader):
        # before train iteration
        self.timer.it_start = time.time()
        # train iteration
        self.optim.zero_grad()
        var = self.graph.forward(opt,var,mode="train")
        loss = self.graph.compute_loss(opt,var,mode="train")
        loss = self.summarize_loss(opt,var,loss)
        loss.all.backward()
        self.optim.step()
        # after train iteration
        if (self.it+1)%opt.freq.scalar==0: self.log_scalars(opt,var,loss,step=self.it+1,split="train")
        if (self.it+1)%opt.freq.vis==0: self.visualize(opt,var,step=self.it+1,split="train")
        self.it += 1
        loader.set_postfix(it=self.it,loss="{:.3f}".format(loss.all))
        self.timer.it_end = time.time()
        util.update_timer(opt,self.timer,self.ep,len(loader))
        return loss

    def summarize_loss(self,opt,var,loss):
        loss_all = 0.
        assert("all" not in loss)
        # weigh losses
        for key in loss:
            assert(key in opt.loss_weight)
            assert(loss[key].shape==())
            if opt.loss_weight[key] is not None:
                assert not torch.isinf(loss[key]),"loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]),"loss {} is NaN".format(key)
                loss_all += 10**float(opt.loss_weight[key])*loss[key]
        loss.update(all=loss_all)
        return loss

    @torch.no_grad()
    def validate(self,opt,ep=None):
        self.graph.eval()
        loss_val = edict()
        loader = tqdm.tqdm(self.test_loader,desc="validating",leave=False)
        for it,batch in enumerate(loader):
            var = edict(batch)
            var = util.move_to_device(var,opt.device)
            var = self.graph.forward(opt,var,mode="val")
            loss = self.graph.compute_loss(opt,var,mode="val")
            loss = self.summarize_loss(opt,var,loss)
            for key in loss:
                loss_val.setdefault(key,0.)
                loss_val[key] += loss[key]*len(var.idx)
            loader.set_postfix(loss="{:.3f}".format(loss.all))
            if it==0: self.visualize(opt,var,step=ep,split="val")
        for key in loss_val: loss_val[key] /= len(self.test_data)
        self.log_scalars(opt,var,loss_val,step=ep,split="val")
        log.loss_val(opt,loss_val.all)

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        for key,value in loss.items():
            if key=="all": continue
            if opt.loss_weight[key] is not None:
                self.tb.add_scalar("{0}/loss_{1}".format(split,key),value,step)
        if metric is not None:
            for key,value in metric.items():
                self.tb.add_scalar("{0}/{1}".format(split,key),value,step)

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train"):
        raise NotImplementedError

    def save_checkpoint(self,opt,ep=0,it=0,latest=False):
        util.save_checkpoint(opt,self,ep=ep,it=it,latest=latest)
        if not latest:
            log.info("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group,opt.name,ep,it))

# ============================ computation graph for forward/backprop ============================

class Graph(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()

    def forward(self,opt,var,mode=None):
        raise NotImplementedError
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        raise NotImplementedError
        return loss

    def L1_loss(self,pred,label=0):
        loss = (pred.contiguous()-label).abs()
        return loss.mean()
    def MSE_loss(self,pred,label=0):
        loss = (pred.contiguous()-label)**2
        return loss.mean()
