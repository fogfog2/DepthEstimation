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
import math 

from depthestimation.utils import readlines
from depthestimation.options_mid import DepthOptions
from depthestimation import datasets, networks
from depthestimation.layers import transformation_from_parameters, disp_to_depth
import tqdm

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = "splits"

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    frames_to_load = [0]
    if opt.use_future_frame:
        frames_to_load.append(1)
    for idx in range(-1, -1 - opt.num_matching_frames, -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    
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
       # filenames = filenames[::2]
        if opt.eval_teacher:
            encoder_path = os.path.join(opt.load_weights_folder, "mono_encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "mono_depth.pth")
            encoder_class = networks.ResnetEncoder

        else:
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

            #encoder_model = "resnet" 
            #encoder_model = "swin_h" 
            encoder_model =  opt.train_model
            
            if not opt.no_teacher:
                if "resnet" in encoder_model:            
                    encoder_class = networks.ResnetEncoderMatching
                elif "swin" in encoder_model:
                    encoder_class = networks.SwinEncoderMatching
                elif "cmt" in encoder_model:
                    encoder_class = networks.CMTEncoderMatching
            else:
                if "resnet" in encoder_model:            
                    encoder_class = networks.ResnetEncoder 
                elif "cmt" in encoder_model:
                    encoder_class = networks.ResnetEncoderCMT
            #encoder_class = networks.ResnetEncoderMatching

        encoder_dict = torch.load(encoder_path)
        try:
            HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width

        img_ext = '.png' if opt.png else '.jpg'
        if opt.eval_split == 'cityscapes':
            dataset = datasets.CityscapesEvalDataset(opt.data_path, filenames,
                                                     HEIGHT, WIDTH,
                                                     frames_to_load, 4,
                                                     is_train=False,
                                                     img_ext=img_ext)
        elif opt.eval_split =='custom_ucl':
            dataset = datasets.UCLRAWDataset(opt.data_path, filenames,
                                                     HEIGHT, WIDTH,
                                                     frames_to_load, 4,
                                                     is_train=False,
                                                     img_ext=img_ext)
        elif opt.eval_split =='custom_mid':
            dataset = datasets.MIDRAWDataset(opt.data_path, filenames,
                                                     HEIGHT, WIDTH,
                                                     frames_to_load, 4,
                                                     is_train=False,
                                                     img_ext=img_ext)
        else:
            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                               encoder_dict['height'], encoder_dict['width'],
                                               frames_to_load, 4,
                                               is_train=False,
                                               img_ext=img_ext)

        
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        # setup models
        if opt.eval_teacher:
            encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False)
        else:
            
            if "resnet" in encoder_model:            
                encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False,
                                input_width=encoder_dict['width'],
                                input_height=encoder_dict['height'],
                                adaptive_bins=True,
                                min_depth_bin=0.1, max_depth_bin=20.0,
                                depth_binning=opt.depth_binning,
                                num_depth_bins=opt.num_depth_bins)
            elif "swin" in encoder_model:
                encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False,
                                input_width=encoder_dict['width'],
                                input_height=encoder_dict['height'],
                                adaptive_bins=True,
                                min_depth_bin=0.1, max_depth_bin=20.0,
                                depth_binning=opt.depth_binning,
                                num_depth_bins=opt.num_depth_bins, use_swin_feature = opt.swin_use_feature)
            elif "cmt" in encoder_model:
                encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False,
                                input_width=encoder_dict['width'],
                                input_height=encoder_dict['height'],
                                adaptive_bins=True,
                                min_depth_bin=0.1, max_depth_bin=20.0,
                                depth_binning=opt.depth_binning,
                                num_depth_bins=opt.num_depth_bins,
                                upconv = opt.cmt_use_upconv, start_layer = opt.cmt_layer, embed_dim = opt.cmt_dim,  use_cmt_feature = opt.cmt_use_feature
                                )
            
            
            pose_enc_dict = torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"))
            pose_dec_dict = torch.load(os.path.join(opt.load_weights_folder, "pose.pth"))

            pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
            pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                            num_frames_to_predict_for=2)

            pose_enc.load_state_dict(pose_enc_dict, strict=True)
            pose_dec.load_state_dict(pose_dec_dict, strict=True)

            min_depth_bin = encoder_dict.get('min_depth_bin')
            max_depth_bin = encoder_dict.get('max_depth_bin')

            pose_enc.eval()
            pose_dec.eval()

            if torch.cuda.is_available():
                pose_enc.cuda(cuda_device)
                pose_dec.cuda(cuda_device)

        encoder = encoder_class(**encoder_opts)                
        
        
        if opt.use_attention_decoder:            
            depth_decoder = networks.DepthDecoderAttention(encoder.num_ch_enc , no_spatial= opt.attention_only_channel)           
        else:
            depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.eval()
        depth_decoder.eval()

        if torch.cuda.is_available():
            encoder.cuda(cuda_device)
            depth_decoder.cuda(cuda_device)

        pred_disps = []
        features_out = []
        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

        # do inference
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                input_color = data[('color', 0, 0)]
                if torch.cuda.is_available():
                    input_color = input_color.cuda()

                if opt.eval_teacher:
                    output = encoder(input_color)
                    output = depth_decoder(output)
                else:

                    if opt.static_camera:
                        for f_i in frames_to_load:
                            data["color", f_i, 0] = data[('color', 0, 0)]

                    # predict poses
                    pose_feats = {f_i: data["color", f_i, 0] for f_i in frames_to_load}
                    if torch.cuda.is_available():
                        pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
                    # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                    for fi in frames_to_load[1:]:
                        if fi < 0:
                            pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                            axisangle, translation = pose_dec(pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=True)

                            # now find 0->fi pose
                            if fi != -1:
                                pose = torch.matmul(pose, data[('relative_pose', fi + 1)])

                        else:
                            pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                            axisangle, translation = pose_dec(pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=False)

                            # now find 0->fi pose
                            if fi != 1:
                                pose = torch.matmul(pose, data[('relative_pose', fi - 1)])

                        data[('relative_pose', fi)] = pose

                    lookup_frames = [data[('color', idx, 0)] for idx in frames_to_load[1:]]
                    lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

                    relative_poses = [data[('relative_pose', idx)] for idx in frames_to_load[1:]]
                    relative_poses = torch.stack(relative_poses, 1)

                    K = data[('K', 2)]  # quarter resolution for matching
                    invK = data[('inv_K', 2)]

                    if torch.cuda.is_available():
                        lookup_frames = lookup_frames.cuda()
                        relative_poses = relative_poses.cuda()
                        K = K.cuda()
                        invK = invK.cuda()

                    if opt.zero_cost_volume:
                        relative_poses *= 0
 
                    if opt.post_process:
                        raise NotImplementedError


                    if not opt.no_teacher:
                        output, lowest_cost, costvol = encoder(input_color, lookup_frames,
                                                            relative_poses,
                                                            K,
                                                            invK,
                                                            min_depth_bin, max_depth_bin)
                    else:
                        output = encoder(input_color)
                    
                    features= output.copy()
                    
                    for i in range(len(features)):
                        features_out.append(features[i].cpu()[0].numpy())
                    
                    output = depth_decoder(output)

                pred_disp, _ = disp_to_depth(output[("disp", 0)],opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)
        print('finished predicting!')

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #delay = round(1000/30.0)
    #out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1216, 352))
    # for idx in range(len(pred_disps)):
    #     disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
    #     depth = np.clip(disp_resized, 0, 10)
    #     dmax, dmin = depth.max(), depth.min()
    #     depth = (depth)/(11)
    #     depth = np.uint8(depth * 256)
    #     #out.write(depth)
    #     cv2.imshow("test", depth)
    #     filename11 = "./image_cmt/" +str(idx).zfill(3) +".png"
    #     cv2.imwrite(filename11, depth)
    #     cv2.waitKey(33)

    if opt.save_pred_disps:
        if opt.zero_cost_volume:
            tag = "zero_cv"
        elif opt.eval_teacher:
            tag = "teacher"
        else:
            tag = "multi"
        output_path = os.path.join(
            opt.load_weights_folder, "{}_{}_split.npy".format(tag, opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.pathfeatures.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()
    # elif opt.eval_split =='custom_ucl':
    #     save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
    #     print("-> Saving out benchmark predictions to {}".format(save_dir))
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    #     for idx in range(len(pred_disps)):
    #         disp_resized = cv2.resize(pred_disps[idx], (256, 256))
    #         depth = 1 / disp_resized
    #         depth = np.clip(depth, 0, 255)
    #         depth = np.uint8(depth * 150)
    #         save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
    #         # cv2.imwrite(save_path, depth)
    #         # cv2.imshow("test", depth)
    #         # cv2.waitKey(1)

    if opt.eval_split == 'cityscapes':
        print('loading cityscapes gt depths individually due to their combined size!')
        gt_depths = os.path.join(splits_dir, opt.eval_split, "gt_depths")
        
    elif opt.eval_split=='custom_ucl':
        gt_path = filenames
        gt_depths = []
        for v in gt_path:
            gt_depths.append(v.replace("FrameBuffer","Depth"))
    elif opt.eval_split=='custom_mid':
        gt_path = filenames
        gt_depths = []
        for v in gt_path:            
            depth_file = v.replace("color_left","depth")
            depth_file = depth_file.replace("JPEG","PNG")        
            gt_depths.append(depth_file)     
        
    else:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    error_1 = []
    error_2 = []
    error_3 = []
    error_4 = []
    ratios = []
    
    counter_1 =0
    counter_2 =0
    counter_3 =0
    counter_4 =0
    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        if opt.eval_split == 'cityscapes':
            gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
            gt_height, gt_width = gt_depth.shape[:2]
            # crop ground truth to remove ego car -> this has happened in the dataloader for input
            # images
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]
        elif opt.eval_split =='custom_ucl':            
            path = os.path.join("/home/sj/colon",gt_depths[i])
            gt_depth =  cv2.imread(path,0)
            gt_height, gt_width = gt_depth.shape[:2]
        elif opt.eval_split =='custom_mid':            
            path = os.path.join("/media/sj/data2/datasets/mid_drone/",gt_depths[i])            
            gt_depth =  cv2.imread(path,0)                                
            gt_height, gt_width = gt_depth.shape[:2]
            
        else:
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = np.squeeze(pred_disps[i])
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == 'cityscapes':
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        elif opt.eval_split == 'cityscapes':
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        else:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
           # mask = gt_depth > 0

        # cmask = np.uint8(crop_mask * 255)
        # save_path = os.path.join("/home/sj/gt_depth/", "{:010d}_m.png".format(i))
        # cv2.imwrite(save_path, cmask)

        # depth = np.clip(gt_depth, 0, 80)        
        # depth = np.uint8(depth * 3)
        # save_path = os.path.join("/home/sj/gt_depth/", "{:010d}.png".format(i))
        # cv2.imwrite(save_path, depth)
        
        # depth2 = np.uint8(pred_depth*100)
        # save_path = os.path.join("/home/sj/gt_depth/", "{:010d}_p.png".format(i))
        # cv2.imwrite(save_path, depth2)

    
        # for idx in range(len(features_out)):            
        #     shape = features_out[idx].shape
            
        #     row = int(math.sqrt(shape[0])+0.5)
        
        set_rowcol = False
        set_sum = True
        
        # if set_rowcol:        
        #     for i in range(5):
                
        #         p = features_out[i].shape[0]
                
        #         for j in range(p):
        #             cv2.normalize(features_out[i][j],features_out[i][j],0,1,cv2.NORM_MINMAX) 
                
        #     step_size = 8
        #     input_channel = 0
        #     img = []
        #     for j in range(step_size):
        #         img1 = np.concatenate((features_out[input_channel][j*step_size+0], features_out[input_channel][j*step_size+1]),axis=1)               
        #         for i in range(1,step_size-1,1):
        #             img1 = np.concatenate((img1, features_out[input_channel][j*step_size+ i+1]),axis=1)            
        #         img.append(img1)
                
        #     img2 = np.concatenate((img[0], img[1]), axis=0)                
        #     for k in range(1,len(img)-1,1):            
        #         img2= np.concatenate((img2, img[k+1]), axis=0)               
        #     #img1=  features_out[0].reshape((1024,1024))        
        #     #cv2.imshow("test",img2)
            
        #     step_size = 8
        #     input_channel = 1
        #     img = []
        #     for j in range(step_size):
        #         img1 = np.concatenate((features_out[input_channel][j*step_size+0], features_out[input_channel][j*step_size+1]),axis=1)               
        #         for i in range(1,step_size-1,1):
        #             img1 = np.concatenate((img1, features_out[input_channel][j*step_size+ i+1]),axis=1)            
        #         img.append(img1)
                
        #     img3 = np.concatenate((img[0], img[1]), axis=0)                
        #     for k in range(1,len(img)-1,1):            
        #         img3= np.concatenate((img3, img[k+1]), axis=0)               
        #     #img1=  features_out[0].reshape((1024,1024))        
        #     #cv2.imshow("test2",img3)
            
        #     if not "cmt" in encoder_model:
        #         step_size = 8
        #         input_channel = 2
        #         img = []
        #         for j in range(step_size*2):
        #             img1 = np.concatenate((features_out[input_channel][j*step_size+0], features_out[input_channel][j*step_size+1]),axis=1)               
        #             for i in range(1,step_size-1,1):
        #                 img1 = np.concatenate((img1, features_out[input_channel][j*step_size+ i+1]),axis=1)            
        #             img.append(img1)
                    
        #         img4 = np.concatenate((img[0], img[1]), axis=0)                
        #         for k in range(1,len(img)-1,1):            
        #             img4= np.concatenate((img4, img[k+1]), axis=0)               
        #         #img1=  features_out[0].reshape((1024,1024))        
        #         #cv2.imshow("test3",img4)
                
        #         step_size = 8
        #         input_channel = 3
        #         img = []
        #         for j in range(step_size*2):
        #             img1 = np.concatenate((features_out[input_channel][j*step_size+0], features_out[input_channel][j*step_size+1]),axis=1)               
        #             for i in range(1,step_size*2-1,1):
        #                 img1 = np.concatenate((img1, features_out[input_channel][j*step_size+ i+1]),axis=1)            
        #             img.append(img1)
                    
        #         img5 = np.concatenate((img[0], img[1]), axis=0)                
        #         for k in range(1,len(img)-1,1):            
        #             img5= np.concatenate((img5, img[k+1]), axis=0)               
        #         #img1=  features_out[0].reshape((1024,1024))        
        #         #cv2.imshow("test4",img5)
        #     else:
        #         step_size = 23
        #         input_channel = 2
        #         img = []
        #         for j in range(4):
        #             img1 = np.concatenate((features_out[input_channel][j*step_size+0], features_out[input_channel][j*step_size+1]),axis=1)               
        #             for i in range(1,step_size-1,1):
        #                 img1 = np.concatenate((img1, features_out[input_channel][j*step_size+ i+1]),axis=1)            
        #             img.append(img1)
                    
        #         img4 = np.concatenate((img[0], img[1]), axis=0)                
        #         for k in range(1,len(img)-1,1):            
        #             img4= np.concatenate((img4, img[k+1]), axis=0)               
        #         #img1=  features_out[0].reshape((1024,1024))        
        #         #cv2.normalize(img4,  img4, 0, 1, cv2.NORM_MINMAX)
        #         #cv2.imshow("test3",img4)
                
        #         step_size = 23
        #         input_channel = 3
        #         img = []
        #         for j in range(8):
        #             img1 = np.concatenate((features_out[input_channel][j*step_size+0], features_out[input_channel][j*step_size+1]),axis=1)               
        #             for i in range(1,step_size-1,1):
        #                 img1 = np.concatenate((img1, features_out[input_channel][j*step_size+ i+1]),axis=1)            
        #             img.append(img1)
                    
        #         img5 = np.concatenate((img[0], img[1]), axis=0)                
        #         for k in range(1,len(img)-1,1):            
        #             img5= np.concatenate((img5, img[k+1]), axis=0)               
        #         #img1=  features_out[0].reshape((1024,1024))        
        #         #cv2.normalize(img5,  img5, 0, 1, cv2.NORM_MINMAX)
        #         #cv2.imshow("test4",img5)
        # else:
        #     for i in range(5):            
        #         p = features_out[i].shape[0]
            
        #         for j in range(p):
        #             cv2.normalize(features_out[i][j],features_out[i][j],0,1,cv2.NORM_MINMAX) 
            
        #     step_size = 8
        #     input_channel = 0
        #     img = []
        #     for j in range(len(features_out[input_channel])):
        #         img.append(features_out[input_channel][j])
            
        #     img2 = sum(img)/len(features_out[input_channel])
        #     #img1=  features_out[0].reshape((1024,1024))        
        #     #cv2.imshow("test",img2)
            
        #     step_size = 8
        #     input_channel = 1
        #     img = []
        #     for j in range(len(features_out[input_channel])):
        #         img.append(features_out[input_channel][j])
            
        #     img3 = sum(img)/len(features_out[input_channel])   
        #     #img1=  features_out[0].reshape((1024,1024))        
        #     #cv2.imshow("test2",img3)
            
        #     if not "cmt" in encoder_model:
        #         step_size = 8
        #         input_channel = 2
        #         img = []
        #         for j in range(len(features_out[input_channel])):
        #             img.append(features_out[input_channel][j])
            
        #         img4 = sum(img)/len(features_out[input_channel])        
        #         #img1=  features_out[0].reshape((1024,1024))        
        #         #cv2.imshow("test3",img4)
                
        #         step_size = 8
        #         input_channel = 3
        #         img = []
        #         for j in range(len(features_out[input_channel])):
        #             img.append(features_out[input_channel][j])
            
        #         img5 = sum(img)/len(features_out[input_channel])     
        #         #img1=  features_out[0].reshape((1024,1024))        
        #         #cv2.imshow("test4",img5)
        #     else:
        #         step_size = 23
        #         input_channel = 2
        #         img = []
        #         for j in range(len(features_out[input_channel])):
        #             img.append(features_out[input_channel][j])
            
        #         img4 = sum(img)/len(features_out[input_channel])  
        #         #img1=  features_out[0].reshape((1024,1024))        
        #         #cv2.normalize(img4,  img4, 0, 1, cv2.NORM_MINMAX)
        #         #cv2.imshow("test3",img4)
                
        #         step_size = 23
        #         input_channel = 3
        #         img = []
        #         for j in range(len(features_out[input_channel])):
        #             img.append(features_out[input_channel][j])
            
        #         img5 = sum(img)/len(features_out[input_channel])  
        #         #img1=  features_out[0].reshape((1024,1024))        
        #         #cv2.normalize(img5,  img5, 0, 1, cv2.NORM_MINMAX)
        #         #cv2.imshow("test4",img5)
        
        # img2 = img2*255
        # img3 = img3*255
        # img4 = img4*255
        # img5 = img5*255
        
        # img2= cv2.applyColorMap(img2.astype(np.uint8), cv2.COLORMAP_JET)
        # img3= cv2.applyColorMap(img3.astype(np.uint8), cv2.COLORMAP_JET)
        # img4= cv2.applyColorMap(img4.astype(np.uint8), cv2.COLORMAP_JET)
        # img5= cv2.applyColorMap(img5.astype(np.uint8), cv2.COLORMAP_JET)
        
        # cv2.imshow("layer1",img2)
        # cv2.imshow("layer2",img3)
        # cv2.imshow("layer3",img4)
        # cv2.imshow("layer4",img5)
        
        # cv2.waitKey(0)
        
        # cv2.imwrite("stage_1.jpg", img2)
        # cv2.imwrite("stage_2.jpg", img3)
        # cv2.imwrite("stage_3.jpg", img4)
        # cv2.imwrite("stage_4.jpg", img5)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]


        #disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
        
        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        #pred_depth_img = cv2.resize(pred_depth, (1242,375))

        # mask1 = gt_depth<20
        # mask2 = np.logical_and(gt_depth>=20, gt_depth <40)
        # mask3 = np.logical_and(gt_depth>=40, gt_depth <60)
        # mask4 = np.logical_and(gt_depth>=60, gt_depth <80)
        
        
        # gt_depth_1 = gt_depth[mask1]
        # gt_depth_2 = gt_depth[mask2]
        # gt_depth_3 = gt_depth[mask3]
        # gt_depth_4 = gt_depth[mask4]
        
        # pred_depth_1 = pred_depth[mask1]
        # pred_depth_2 = pred_depth[mask2]
        # pred_depth_3 = pred_depth[mask3]
        # pred_depth_4 = pred_depth[mask4]
        
        # counter_1 +=len(pred_depth_1)
        # counter_2 +=len(pred_depth_2)
        # counter_3 +=len(pred_depth_3)
        # counter_4 +=len(pred_depth_4)
        
        # error_1.append(compute_errors(gt_depth_1, pred_depth_1))
        # if len(pred_depth_2) !=0:
        #     error_2.append(compute_errors(gt_depth_2, pred_depth_2))
        
        # if len(pred_depth_3) !=0:
        #     error_3.append(compute_errors(gt_depth_3, pred_depth_3))
        # if len(pred_depth_4) !=0:
        #     error_4.append(compute_errors(gt_depth_4, pred_depth_4))
        
        errors.append(compute_errors(gt_depth, pred_depth))

    if opt.save_pred_disps:
        print("saving errors")
        if opt.zero_cost_volume:
            tag = "mono"
        else:
            tag = "multi"
        output_path = os.path.join(
            opt.load_weights_folder, "{}_{}_errors.npy".format(tag, opt.eval_split))
        np.save(output_path, np.array(errors))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    
    # mean_errors_1 = np.array(error_1).mean(0)
    # mean_errors_2 = np.array(error_2).mean(0)
    # mean_errors_3 = np.array(error_3).mean(0)
    # mean_errors_4 = np.array(error_4).mean(0)
    
   # print(counter_1, counter_2,counter_3,counter_4)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                           "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    # print(("&{: 8.3f}  " * 7).format(*mean_errors_1.tolist()) + "\\\\")
    # print(("&{: 8.3f}  " * 7).format(*mean_errors_2.tolist()) + "\\\\")
    # print(("&{: 8.3f}  " * 7).format(*mean_errors_3.tolist()) + "\\\\")
    # print(("&{: 8.3f}  " * 7).format(*mean_errors_4.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = DepthOptions()
    evaluate(options.parse())
