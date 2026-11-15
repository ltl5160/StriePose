from calendar import c
import os
from re import S
import sys

from sympy import im, use

sys.path.insert(0, os.getcwd())

from config_parser import parse_cfg
import argparse

from tools_for_BOP import bop_io
from tools_for_BOP.common_dataset_info import get_obj_info
from bop_dataset_pytorch import bop_dataset_single_obj_pytorch

import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np

from binary_code_helper.CNN_output_to_pose import load_dict_class_id_3D_points, CNN_outputs_to_object_pose

sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout
from model.BinaryCodeNet import BinaryCodeNet_Deeplab
from model.BinaryCodeNet import MaskLoss, BinaryCodeLoss, SymLossWeighted
from model.mobilenetv3.segmentation import MobileNetV3Seg
from model.RepViTNet import RepViTNet


from tensorboardX import SummaryWriter

from utils import save_checkpoint, get_checkpoint, save_best_checkpoint, save_image
from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP

from get_detection_results import get_detection_results, ycbv_select_keyframe

from common_ops import from_output_to_class_mask, from_output_to_class_binary_code, get_batch_size

from test_network_with_test_data import test_network_with_single_obj
from bop_toolkit_lib import misc

import os
os.environ['CUDA_VISIBLE_DEVICES'] ="0"

def load_partial_weights(model, checkpoint):
    model_dict = model.state_dict()
    target_key = ['net.aspp.conv_1x1_4.weight','net.aspp.conv_1x1_4.bias']
    for key in checkpoint.keys():
        if key in model_dict:
            if key in target_key:
                pretrained_weight = checkpoint[key]
                current_weight = model_dict[key]
                if pretrained_weight.shape[1:] == current_weight.shape[1:]:
                    pre_w = pretrained_weight.shape[0]
                    cur_w = current_weight.shape[0]
                    # 新建一个与当前模型权重同形状的张量，并初始化
                    new_weight = current_weight.clone()
                    # 将预训练权重复制到前128个通道
                    new_weight[:pre_w] = pretrained_weight
                    # 剩余通道保持当前初始化（或自定义初始化）
                    # 例如，使用正态分布初始化新增部分
                    import torch.nn as nn
                    nn.init.normal_(new_weight[pre_w:], mean=0, std=0.01)
                    
                    # 更新模型参数
                    model_dict[key] = new_weight

            else:
                model_dict[key] = checkpoint[key]
    # 更新模型参数
    model.load_state_dict(model_dict)
    return model

def main(configs):
    torch.manual_seed(0)      # the both are only good for ablation study
    np.random.seed(0)         # if can be removed in the final experiments

    config_file_name = configs['config_file_name']
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    training_data_folder=configs['training_data_folder']
    training_data_folder_2=configs['training_data_folder_2']
    val_folder=configs['val_folder']                                  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']              # the percentage of second dataset in the batch
    num_workers = configs['num_workers']                                # for data loader
    train_obj_visible_theshold = configs['train_obj_visible_theshold']  # for test is always 0.1, for training we can set different values
    #### network settings
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']        # network output size
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']              # now only support "L1" or "BCE"
    mask_binary_code_loss=configs['mask_binary_code_loss']          # if binary code loss only applied for object mask region
    use_histgramm_weighted_binary_loss = configs['use_histgramm_weighted_binary_loss']
    output_kernel_size = configs['output_kernel_size']                  # last layer kernel size
    resnet_layer = configs['resnet_layer']                              # usually resnet 34
    concat=configs['concat_encoder_decoder']  
    predict_entire_mask=configs['predict_entire_mask']                  # if predict the entire object part rather than the visible one
    if 'efficientnet_key' in configs.keys():
        efficientnet_key = configs['efficientnet_key']
        # print(efficientnet_key)
    else:
        efficientnet_key = None
    # print(efficientnet_key)
    #### pretrain points
    load_pretrain_checkpoint = configs['load_pretrain_checkpoint']
    pretrain_path = configs['pretrain_path']   

    #### check points
    load_checkpoint = configs['load_checkpoint']
    tensorboard_path = configs['tensorboard_path']
    check_point_path = configs['check_point_path']
    image_path = configs['image_path']
    total_epoch = configs['total_epoch']                         # train how many steps
    #### optimizer
    optimizer_type = configs['optimizer_type']                           # Adam is the best sofar
    batch_size=configs['batch_size']                                     # 32 is the best so far, set to 16 for debug in local machine
    learning_rate = configs['learning_rate']                             # 0.002 or 0.003 is the best so far
    binary_loss_weight = configs['binary_loss_weight']                     # 3 is the best so far
    #### augmentations
    Detection_reaults=configs['Detection_reaults']                       # for the test, the detected bounding box provided by GDR Net
    padding_ratio=configs['padding_ratio']                               # pad the bounding box for training and test
    resize_method = configs['resize_method']                             # how to resize the roi images to 256*256
    use_peper_salt= configs['use_peper_salt']                            # if add additional peper_salt in the augmentation
    use_motion_blur= configs['use_motion_blur']                          # if add additional motion_blur in the augmentation
    # vertex code settings
    divide_number_each_itration = configs['divide_number_each_itration']
    number_of_itration = configs['number_of_itration']

    sym_aware_training=configs['sym_aware_training']


    #############build the network 
    binary_code_length = number_of_itration
    print("binary_code_length: ", binary_code_length)
    configs['binary_code_length'] = binary_code_length


    model = RepViTNet(
                binary_code_length=binary_code_length,
                model_name='repvit_m1_1.dist_300e_in1k',
                pretrained=True,
        )
    # model = BinaryCodeNet_Deeplab(
    #                 num_resnet_layers=resnet_layer, 
    #                 concat=concat, 
    #                 binary_code_length=binary_code_length, 
    #                 divided_number_each_iteration = divide_number_each_itration, 
    #                 output_kernel_size = output_kernel_size,
    #                 efficientnet_key = efficientnet_key,
    #                 sym_input_training=False
    #             )
    model.cuda()

    model.eval()

    x = torch.randn(1, 3, 256, 256).cuda()

    with torch.no_grad():

        # features = model.backbone(x)

        # print("Backbone:", features[0].shape)

        mask, binary_code = model(x)

        print("Mask:", mask.shape)
        print("Binary:", binary_code.shape)
            
    import time

    # =====================================
    # 3. 正式测试推理时间
    # =====================================
    iterations = 500
    times = []

    with torch.inference_mode():

        for _ in range(iterations):

            torch.cuda.synchronize()
            start = time.perf_counter()

            mask, binary_code = model(x)

            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)


    # =====================================
    # 4. 统计结果
    # =====================================
    times = np.array(times)

    mean_time = np.mean(times)
    median_time = np.median(times)
    std_time = np.std(times)
    min_time = np.min(times)
    p95_time = np.percentile(times, 95)

    fps = 1000.0 / mean_time

    print("\n========== Inference Speed ==========")
    print("Input Size:       1 x 3 x 256 x 256")
    print("Iterations:       ", iterations)
    print("Mean Latency:     {:.3f} ms".format(mean_time))
    print("Median Latency:   {:.3f} ms".format(median_time))
    print("Std Latency:      {:.3f} ms".format(std_time))
    print("Min Latency:      {:.3f} ms".format(min_time))
    print("P95 Latency:      {:.3f} ms".format(p95_time))
    print("FPS:              {:.2f}".format(fps))
    print("====================================")


    # =====================================
    # 5. GPU显存
    # =====================================
    memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
    memory_reserved = torch.cuda.max_memory_reserved() / 1024**2

    print("\n========== GPU Memory ==========")
    print("Allocated: {:.2f} MB".format(memory_allocated))
    print("Reserved:  {:.2f} MB".format(memory_reserved))
    print("================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str) # config file
    parser.add_argument('--obj_name', type=str) # config file
    # parser.add_argument('--sym_aware_training', type=str, choices=('True','False'), default='False') # config file
    parser.add_argument('--use_cuda', type=str, choices=('True','False'), default='True') # config file
    args = parser.parse_args()
    config_file = args.cfg
    configs = parse_cfg(config_file)
    configs['obj_name'] = args.obj_name
    # configs['sym_aware_training'] = (args.sym_aware_training == 'True')
    configs['use_cuda'] = (args.use_cuda == 'True')
    

    check_point_path = configs['check_point_path']
    # tensorboard_path= configs['tensorboard_path']
    # image_path = configs['image_path']
    config_file_name = os.path.basename(config_file)
    config_file_name = os.path.splitext(config_file_name)[0]
    check_point_path = check_point_path + config_file_name
    # tensorboard_path = tensorboard_path + config_file_name
    configs['check_point_path'] = check_point_path +'/'+ args.obj_name + '/checkpoints/'
    configs['tensorboard_path'] = check_point_path +'/'+ args.obj_name + '/logdir/'
    configs['image_path'] = check_point_path +'/'+ args.obj_name + '/image/'
    configs['config_file_name'] = config_file_name

    if configs['Detection_reaults'] != 'none':
        Detection_reaults = configs['Detection_reaults']
        dirname = os.path.dirname(__file__)
        Detection_reaults = os.path.join(dirname, Detection_reaults)
        configs['Detection_reaults'] = Detection_reaults

    #print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)

    main(configs)