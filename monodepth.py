from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil

import torch
from torchvision import transforms

import matplotlib as mpl
import matplotlib.cm as cm

import monodepth2.layers
from monodepth2 import networks
from monodepth2.layers import disp_to_depth
from monodepth2.utils import download_model_if_doesnt_exist

class Monodepth:

    def __init__(self, model_name, verbose = False):
        self.__verbose = verbose

        self.model_name = model_name
        self.device = torch.device("cpu")

        self.load_model()

    def __print_verbose(self, *args):
        if self.__verbose:
            print(*args)

    def load_model(self):
        def load_encoder(model_path):
            self.__print_verbose("   Loading pretrained encoder")
            encoder_path = os.path.join(model_path, "encoder.pth")

            encoder = networks.ResnetEncoder(18, False)
            loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
            encoder.load_state_dict(filtered_dict_enc)
            encoder.to(self.device)
            encoder.eval()

            return encoder, loaded_dict_enc['width'], loaded_dict_enc['height']

        def load_depth_decoder(model_path, encoder):
            self.__print_verbose("   Loading pretrained decoder")
            depth_decoder_path = os.path.join(model_path, "depth.pth")
        
            depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
            loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
            
            depth_decoder.load_state_dict(loaded_dict)
            depth_decoder.to(self.device)
            depth_decoder.eval()

            return depth_decoder

        download_model_if_doesnt_exist(self.model_name)

        model_path = os.path.join("models", self.model_name)
        self.__print_verbose("-> Loading model from ", model_path)
        
        self.encoder, self.feed_width, self.feed_height = load_encoder(model_path)
        self.depth_decoder = load_depth_decoder(model_path, self.encoder)
        

    def parse_frame(self, frame, save_to = None):
        # Load image and preprocess
        original_width, original_height = frame.size
        input_image = frame.resize((self.feed_width, self.feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(self.device)
        with torch.no_grad():
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Get depth map as np array
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        disparity = scaled_disp.cpu().numpy() # arg to return

        # Get colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        if save_to is not None:
            if not os.path.isdir(save_to):
                os.mkdir(save_to, 0o755)
            
            im.save(os.path.join(save_to, "depth.png"))
            np.save(os.path.join(save_to, "disparity.npy"), disparity)

        return im, disparity

if __name__ == "__main__":
    model = Monodepth("mono_640x192")
    im = pil.open("monodepth2/assets/test_image.jpg").convert('RGB')

    im, disparity = model.parse_frame(im, save_to="output")