# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import skimage.transform
import numpy as np
import PIL.Image as pil
import csv
import pandas as pd


from depthestimation.kitti_utils import generate_depth_map
from .dummy_mono_dataset import DUMMYMonoDataset


class DUMMYDataset(DUMMYMonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(DUMMYDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.full_res_shape = (256, 256)

    def load_data_path(self,index):
        line = self.filenames[index].split()
        img_path = line[0]
        label_file = "labels/"+img_path[7:17]+".csv"
        label_num = int(img_path[-8:-4])
        return img_path, label_file, label_num

    
    def get_color(self, folder, do_flip):
        color = self.loader(self.get_image_path(folder))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_target_pt(self, label_path,  label_num):
        csv_path = os.path.join(self.data_path, label_path) 
        csv_file  = pd.read_csv(csv_path)
        
        px = csv_file.iloc[:,2].values[label_num]
        py = csv_file.iloc[:,3].values[label_num]
        cls = csv_file.iloc[:,4].values[label_num]
        return px, py, cls

class DUMMYRAWDataset(DUMMYDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(DUMMYRAWDataset, self).__init__(*args, **kwargs)


    def get_image_path(self, folder):
        image_path = os.path.join(self.data_path, folder)        
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        image_path = os.path.join(self.data_path, folder)
        base, ext = os.path.splitext(os.path.basename(folder))
        f_idx = int(base[-4:])+frame_index

        depth_path = os.path.join(self.data_path,  os.path.dirname(folder), "Depth_"+str(f_idx).zfill(4) + ext)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

