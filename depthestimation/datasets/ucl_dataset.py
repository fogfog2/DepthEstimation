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

from depthestimation.kitti_utils import generate_depth_map
from .ucl_mono_dataset import UCLMonoDataset


class UCLDataset(UCLMonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(UCLDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[0.3468, 0, 0.4992, 0],
                           [0, 0.3544, 0.4978, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (256, 256)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0]

        depth_file = line.replace("FrameBuffer","Depth")
        #scene_name = line[0]
        #frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            depth_file)

        return os.path.isfile(velo_filename)

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()        
        
        tt = os.path.basename(line[0])
        base, ext = os.path.splitext(tt)
        if ext == '.png':
            frame_index = int(base[-4:])
        folder = os.path.dirname(line[0])
        
        return folder, frame_index
    
    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class UCLRAWDataset(UCLDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(UCLRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):

        image_path = os.path.join(self.data_path, folder)
        #base, ext = os.path.splitext(os.path.basename(folder))
        #f_idx = int(base[-4:])+frame_index
        path = os.path.join(image_path, "FrameBuffer_"+str(frame_index).zfill(4)+".png")
        #new
        #path = os.path.join(self.data_path, folder)
        
        #synthetic
        #path = os.path.join(self.data_path,  os.path.dirname(folder), str(f_idx).zfill(4) + ext)
        return path

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

