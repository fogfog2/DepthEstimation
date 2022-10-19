# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
import time
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import readlines, sec_to_hm_str
from layers import compute_rmse, compute_depth_errors, BCELoss

from depthestimation import datasets, networks
import matplotlib.pyplot as plt
datasets
from collections import OrderedDict

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        cuda_device = "cuda:"+str(self.opt.cuda_device)
        print(cuda_device)
        self.device = torch.device(cuda_device)

        # MODEL SETUP

        encoder_model = self.opt.train_model 
        #encoder_model = "swin_h" 
        #encoder_model = "cmt_h"
        self.bceLoss = BCELoss()
        self.bceLoss.to(self.device)
    
        if "resnet" in encoder_model:            
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained",
                input_height=self.opt.height, input_width=self.opt.width,
                adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
                depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins)
        elif "cmt" in encoder_model:            
            self.models["encoder"] = networks.ResnetEncoderCMT(
                self.opt.num_layers, self.opt.weights_init == "pretrained",
                input_height=self.opt.height, input_width=self.opt.width,
                adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
                depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins, upconv = self.opt.cmt_use_upconv, start_layer = self.opt.cmt_layer, embed_dim = self.opt.cmt_dim, use_cmt_feature = self.opt.cmt_use_feature)


        self.models["encoder"].to(self.device)

        if self.opt.use_attention_decoder:            
            self.models["decoder"] = networks.TargetDecoderAttention(
                self.models["encoder"].num_ch_enc, self.opt.scales , no_spatial= self.opt.attention_only_channel)            
        else:
            self.models["decoder"] = networks.TargetDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)

        self.models["decoder"].to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["decoder"].parameters())
        
        if self.opt.use_adamw:
            self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
        else:
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, self.opt.scheduler_step_ratio)
        
        if self.opt.load_weights_folder is not None:            
                self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # DATA
        datasets_dict = {"custom_dummy": datasets.DUMMYRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
             is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
             is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
                m.train()
                

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs, is_train=True)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            outputs['lr'] = self.model_optimizer.param_groups[0]['lr']
            
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()             

            if self.opt.save_intermediate_models and late_phase:
                self.save_model(save_step=True)

            
            self.step += 1            
            
        print("step: {}, lr: {}".format(self.step,outputs['lr']))            
        self.model_lr_scheduler.step()

    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        features = self.models["encoder"](inputs["color_resize_aug"])            
        outputs.update(self.models["decoder"](features))

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

            
    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0
        
        ## calcuate loss
  
        #diff_px = abs ( inputs[("px_norm")] - outputs[("out_xyc")][:,0] )
        #diff_py = abs ( inputs[("py_norm")] - outputs[("out_xyc")][:,1] )
        #diff_cls = abs ( inputs[("cls")] - outputs[("out_xyc")][:,2] ) 
        #loss = diff_px+diff_py +diff_cls
        
        px_loss = compute_rmse(inputs[("px_norm")] , outputs[("out_xyc")][:,0])
        py_loss = compute_rmse(inputs[("py_norm")] , outputs[("out_xyc")][:,1])

        cls_loss = self.bceLoss(inputs[("cls")] , outputs[("out_xyc")][:,0]) *0.05  
        loss = px_loss+py_loss+cls_loss
        tatal_loss = loss.mean()


        ##
        losses["loss"] = tatal_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        
        log_path = self.log_path+"/log.txt"
        f = open(log_path,'a')        
        
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())
            f.write(depth_errors[i].cpu() + ',')
        
        f.write('\n')
        f.close()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        
        

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        if outputs.get("lr") is not None:
            writer.add_scalar("lr", outputs["lr"], self.step)

        ## add log image

       # disp = colormap(outputs[("disp"),0][0,0])
        # writer.add_image(
        #     "disp_",
        #     disp, self.step)
        # ## ~

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, save_step=False):
        """Save model weights to disk
        """        
        if save_step:
            save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,
                                                                                       self.step))
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                # save estimates of depth bins

            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


def same_shape(shape1, shape2):
    """
    Checks if two shapes are the same

    Parameters
    ----------
    shape1 : tuple
        First shape
    shape2 : tuple
        Second shape

    Returns
    -------
    flag : bool
        True if both shapes are the same (same length and dimensions)
    """
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True

def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
