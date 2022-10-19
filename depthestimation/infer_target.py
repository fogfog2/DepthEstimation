# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from depthestimation.utils import readlines
from depthestimation.options_dummy_model import DepthOptions
from depthestimation import datasets, networks
import tqdm

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = "splits"


def inference(opt):
    """Evaluates a pretrained model using a specified test set
    """

    if opt.cuda_device is None:
        cuda_device = "cuda:0"
    else:
        cuda_device = opt.cuda_device

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # Setup dataloaders
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "decoder.pth")

        #encoder_model = "resnet" 
        #encoder_model = "swin_h" 
        encoder_model =  opt.train_model

        if "resnet" in encoder_model:            
            encoder_class = networks.ResnetEncoder 
        encoder_dict = torch.load(encoder_path)

        try:
            HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width

        img_ext = '.png' if opt.png else '.jpg'
        dataset = datasets.DUMMYRAWDataset(opt.data_path, filenames,
                                                     HEIGHT, WIDTH,
                                                     is_train=False,
                                                     img_ext=img_ext)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        # setup models
        
        if "resnet" in encoder_model:            
            encoder_opts = dict(num_layers=opt.num_layers,
                            pretrained=False,
                            input_width=encoder_dict['width'],
                            input_height=encoder_dict['height'],
                            adaptive_bins=True,
                            min_depth_bin=0.1, max_depth_bin=20.0,
                            depth_binning=opt.depth_binning,
                            num_depth_bins=opt.num_depth_bins)

        encoder = encoder_class(**encoder_opts)                
        
        
        target_decoder = networks.TargetDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        target_decoder.load_state_dict(torch.load(decoder_path))

        encoder.eval()
        target_decoder.eval()

        if torch.cuda.is_available():
            encoder.cuda(cuda_device)
            target_decoder.cuda(cuda_device)

        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

        # do inference
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                input_color = data[('color_resize')]
                if torch.cuda.is_available():
                    input_color = input_color.cuda()

                output = encoder(input_color)
                output = target_decoder(output)

                color_img = input_color.cpu()[0].numpy().transpose(1,2,0)
                color_img = (color_img*255).astype(np.uint8)
                
                #color_img = color_img[:, :, ::-1]
                color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

                gt_px = int(data[('px_norm')].numpy()[0]*255)
                gt_py = int(data[('py_norm')].numpy()[0]*255)
                gt_cls = data[('cls')].numpy()[0]
                t = output[('out_xyc')].cpu().numpy()
                pred_x = int(t[0]*255)
                pred_y = int(t[1]*255)
                #pred_cls= int(t[2]*255)
                pred_cls = 0
                if t[2]>0.5:
                    pred_cls =255

                cv2.circle(color_img, (gt_px,gt_py), 5, (255,0,0))
                cv2.circle(color_img, (pred_x,pred_y), 3, (0,255,0))

                cv2.circle(color_img, (128,128), 120, (pred_cls,pred_cls,pred_cls))
                #color_img= cv2.circle(color_img, (int(55),int(55)), 3, (0,255,0))
                cv2.imshow("test", color_img)
                cv2.waitKey(33)


if __name__ == "__main__":
    options = DepthOptions()
    inference(options.parse())
