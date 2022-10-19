import cv2
import numpy as np
import torch
import os
#from options_ucl import DepthOptions
from layers import transformation_from_parameters, disp_to_depth
import networks
from torchvision import transforms

from hole import draw_hole
from depthestimation.options_ucl import DepthOptions

class Inference():
    
    def __init__(self, opt, model_dir, infer_mono=True, train_model="resnet"):

        opt.load_weights_folder=model_dir
        opt.eval_mono = infer_mono
        #opt.no_teacher = True
        opt.png = True
        opt.train_model =train_model

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        if opt.cuda_device is None:
            cuda_device = "cuda:0"
        else:
            cuda_device = opt.cuda_device
        
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))
        

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_model =  opt.train_model
        
        if not opt.eval_mono:
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
                
        encoder_dict = torch.load(encoder_path)
                
        try:
                HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                    'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width
        
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

        self.pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
        self.pose_dec = networks.PoseDecoder(self.pose_enc.num_ch_enc, num_input_features=1,
                                        num_frames_to_predict_for=2)

        self.pose_enc.load_state_dict(pose_enc_dict, strict=True)
        self.pose_dec.load_state_dict(pose_dec_dict, strict=True)

        self.min_depth_bin = encoder_dict.get('min_depth_bin')
        self.max_depth_bin = encoder_dict.get('max_depth_bin')

        self.pose_enc.eval()
        self.pose_dec.eval()
        
        if torch.cuda.is_available():
            self.pose_enc.cuda(cuda_device)
            self.pose_dec.cuda(cuda_device)
        
        self.encoder = encoder_class(**encoder_opts)     
        
        if opt.use_attention_decoder:            
            self.depth_decoder = networks.DepthDecoderAttention(self.encoder.num_ch_enc , no_spatial= opt.attention_only_channel)           
        else:
            self.depth_decoder = networks.DepthDecoder(self.encoder.num_ch_enc)
        
        model_dict = self.encoder.state_dict()
        self.encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        self.depth_decoder.load_state_dict(torch.load(decoder_path))

        self.encoder.eval()
        self.depth_decoder.eval()
        
        if torch.cuda.is_available():
            self.encoder.cuda(cuda_device)
            self.depth_decoder.cuda(cuda_device)
        
    
    def inference(self, opt,  input_color):       
            
        with torch.no_grad():            
            output = self.encoder(input_color)
            output = self.depth_decoder(output)                
            pred_disp, pred_depth= disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_depth = pred_depth.cpu()[:, 0].numpy()
            pred_disp = pred_disp.cpu()[:, 0].numpy()
        return pred_disp, pred_depth

    def inference_many(self, opt,  input_color, input_prev, pose, K, invK):       
            
        with torch.no_grad():
            lookup_frames = input_prev
            relative_poses = pose
        
            output,_ ,_ = self.encoder(input_color, lookup_frames,
                            relative_poses,
                            K,
                            invK,
                            self.min_depth_bin, self.max_depth_bin)
            output = self.depth_decoder(output)    
            
            pred_disp, pred_depth= disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_depth = pred_depth.cpu()[:, 0].numpy()
            pred_disp = pred_disp.cpu()[:, 0].numpy()
        return pred_disp, pred_depth

    def pose_inference(self, opt,  input_prev, input_current):       
            
        with torch.no_grad():            
            pose_inputs = [input_prev, input_current]
            output = [self.pose_enc(torch.cat(pose_inputs,1))]
            axisangle, translation = self.pose_dec(output)    
            pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
            
        return pose

if __name__ =="__main__":
    options = DepthOptions()
    parsed_option = options.parse()

    model_dir = "/home/sj/tmp/mono_colon_loss2/mdp/models/weights_39"
    #model_dir = "/home/sj/tmp/mono_colon_loss/mdp/models/weights_39"
    #model_dir = "/home/sj/Downloads/colon_model/model_drl"
    #train_model = "cmt"
    train_model = "resnet"
    load_weights_folder = model_dir
    eval_mono = True
    if(len(os.listdir(model_dir)) >6):
        eval_mono = False

    #many

    WIDTH = 256 
    HEIGHT =256
    K = np.array([[0.3468*WIDTH, 0, 0.4992*WIDTH, 0],
                        [0, 0.3544*HEIGHT, 0.4970*HEIGHT, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)

    invk = np.linalg.pinv(K)
    invk = torch.Tensor(invk).unsqueeze(0)
    cuda_invk = invk.cuda()
    cuda_K = torch.Tensor(K).unsqueeze(0).cuda()

    infer = Inference(parsed_option,model_dir,infer_mono=eval_mono, train_model=train_model)            
    to_tensor = transforms.ToTensor()
 
    INPUT_MODE = ["VIDEO","IMAGE"]
    
    Mode = INPUT_MODE[1]
    
    if Mode==INPUT_MODE[0]:
        cap = cv2.VideoCapture(0)
    else:
        img_dir = "/home/sj/exp_20220322-190049"
        filenames = sorted(os.listdir(img_dir))
    
    count = 0
    prev_image =[]
    while True:                
        if Mode==INPUT_MODE[0]:
            ret, cv_image = cap.read()
        else:                        
            if len(filenames)<=count:
                count = 0
            img_path = os.path.join(img_dir,filenames[count])            
            cv_image = cv2.imread(img_path)
            cv_image=  cv2.resize(cv_image,(256,256))
            count=count+1            
                    
        tensor_image =  to_tensor(cv_image).unsqueeze(0)        
        
        if torch.cuda.is_available():
            tensor_image = tensor_image.cuda()

        
        if len(prev_image)>0 and not eval_mono:
            prev_img = prev_image.pop()
            tensor_pose = infer.pose_inference(parsed_option, prev_img, tensor_image)
        elif not eval_mono:
            prev_image.append(tensor_image)
            continue

        if eval_mono:
            disp, depth = infer.inference(parsed_option,tensor_image)
        else:
            prev_img= prev_img.unsqueeze(1)
            tensor_pose =tensor_pose.unsqueeze(1)
            disp, depth = infer.inference_many(parsed_option,tensor_image,
                                                            prev_img, 
                                                            tensor_pose,
                                                            cuda_K,
                                                            cuda_invk)

        prev_image.append(tensor_image)

        sacled_depth = np.clip( (np.squeeze(depth)*180) ,0,255).astype(np.uint8)
        sacled_disp = np.clip((np.squeeze(disp)*25),0,255).astype(np.uint8)
        

        color = cv2.applyColorMap(sacled_depth, cv2.COLORMAP_MAGMA)
        disp_color = cv2.applyColorMap(sacled_disp, cv2.COLORMAP_MAGMA)
         
        x,y  = draw_hole(sacled_disp, cv_image)
        cv_image = cv2.circle( cv_image, (x,y), 5,(255,0,0))

        cv2.imshow("input",cv_image)        
        cv2.imshow("output",color)
        cv2.imshow("output_disp",disp_color)
        cv2.waitKey(0)
        