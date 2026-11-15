from calendar import c
from distutils.command.config import config
import os
import sys

sys.path.insert(0, os.getcwd())

from config_parser import parse_cfg
import argparse
import shutil
from tqdm import tqdm


from tools_for_BOP import bop_io
from bop_dataset_pytorch import bop_dataset_single_obj_pytorch

import torch
import numpy as np
import cv2

from binary_code_helper.CNN_output_to_pose import load_dict_class_id_3D_points, CNN_outputs_to_object_pose
sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout

from model.BinaryCodeNet import BinaryCodeNet_Deeplab

from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP

from get_detection_results import ycbv_select_keyframe
from common_ops import from_output_to_class_mask, from_output_to_class_binary_code, compute_original_mask, calc_prediction_confidence_image
from tools_for_BOP.common_dataset_info import get_obj_info

from binary_code_helper.generate_new_dict import generate_new_corres_dict

from tools_for_BOP import write_to_cvs 

def main(configs):
    output_kernel_size = configs['output_kernel_size']                  # last layer kernel size
    resnet_layer = configs['resnet_layer']                              # usually resnet 34
    concat=configs['concat_encoder_decoder']   
    if 'efficientnet_key' in configs.keys():
        efficientnet_key = configs['efficientnet_key']      
    else  :
        efficientnet_key = None     
    if 'sym_input_training' in configs.keys():
        sym_input_training = configs['sym_input_training']
    else:
        sym_input_training = False

    # pixel code settings
    divide_number_each_itration = configs['divide_number_each_itration']
    number_of_itration = configs['number_of_itration']



    torch.manual_seed(0)      # the both are only good for ablation study
    np.random.seed(0)         # if can be removed in the final experiments

 
    
    divide_number_each_itration = int(divide_number_each_itration)
    number_of_itration = int(number_of_itration)

 
    binary_code_length = number_of_itration
    print("predicted binary_code_length", binary_code_length)
    configs['binary_code_length'] = binary_code_length
 
    net = BinaryCodeNet_Deeplab(
                num_resnet_layers=resnet_layer, 
                concat=concat, 
                binary_code_length=binary_code_length, 
                divided_number_each_iteration = divide_number_each_itration, 
                output_kernel_size = output_kernel_size,
                efficientnet_key = efficientnet_key,
                sym_input_training=sym_input_training
            )


    checkpoint = torch.load( configs['checkpoint_file'] )
    if 'pth' in configs['checkpoint_file']:
        net.load_state_dict(checkpoint)
    else:
        net.load_state_dict(checkpoint['model_state_dict'])

    torch.save(net, os.path.join(configs['eval_output_path'], configs['obj_name']+"_pose_estimation.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str) # config file
    parser.add_argument('--obj_name', type=str)
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--output_path', type=str)


    args = parser.parse_args()
    config_file = args.cfg
    checkpoint_file = args.ckpt_file
    output_path = args.output_path
    obj_name = args.obj_name
    configs = parse_cfg(config_file)

    configs['obj_name'] = obj_name


    configs['checkpoint_file'] = checkpoint_file
    configs['eval_output_path'] = output_path



    main(configs)
