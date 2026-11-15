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

    if 'sym_aware_training' in configs.keys():
        sym_aware_training=configs['sym_aware_training']
    else:
        sym_aware_training=False

    # get dataset informations
    dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global, cam_params = bop_io.get_dataset(bop_path,dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1)    # now the obj_id started from 0
    if obj_name in symmetry_obj:
        sym_bit = symmetry_obj[obj_name]
        if 'sym_input_training' in configs.keys():
            sym_input_training = configs['sym_input_training']
        else:
            sym_input_training = False
        if 'sym_aware_loss' in configs.keys():
            use_sym_aware_loss = configs['sym_aware_loss']
        else:
            use_sym_aware_loss = False
    else:  
        sym_input_training = False
        use_sym_aware_loss = False
        
        
    if 'backbone_name' in configs.keys():
        backbone_name = configs['backbone_name']
    else:
        backbone_name = 'resnet'


    mesh_path = model_plys[obj_id+1]         # mesh_path is a dict, the obj_id should start from 1
    print(mesh_path, flush=True)
    obj_diameter = model_info[str(obj_id+1)]['diameter']

    print("obj_diameter", obj_diameter, flush=True)
    path_dict = os.path.join(dataset_dir, "models_GT_color", "Class_CorresPoint{:06d}.txt".format(obj_id+1))
    total_numer_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    divide_number_each_itration = int(divide_number_each_itration)
    total_numer_class = int(total_numer_class)
    number_of_itration = int(number_of_itration)
    if divide_number_each_itration ** number_of_itration != total_numer_class:
        raise AssertionError("the combination is not valid")
    if BoundingBox_CropSize_image / BoundingBox_CropSize_GT != 2:
        raise AssertionError("currnet endoder-decoder only support input_size/output_size = 2")
    GT_code_infos = [divide_number_each_itration, number_of_itration, total_numer_class]

    if divide_number_each_itration != 2 and (BinaryCode_Loss_Type=='BCE' or BinaryCode_Loss_Type=='L1'):
        raise AssertionError("for non-binary case, use CE as loss function")
    if divide_number_each_itration == 2 and BinaryCode_Loss_Type=='CE':
        raise AssertionError("not support for now")

    vertices = inout.load_ply(mesh_path)["pts"]

    ########################## define data loader
    batch_size_1_dataset, batch_size_2_dataset = get_batch_size(second_dataset_ratio, batch_size)

    train_dataset = bop_dataset_single_obj_pytorch(
                                                    dataset_dir, training_data_folder, rgb_files[obj_id], mask_files[obj_id], mask_visib_files[obj_id], 
                                                    gts[obj_id], gt_infos[obj_id], cam_params[obj_id], True, BoundingBox_CropSize_image, 
                                                    BoundingBox_CropSize_GT, GT_code_infos,  padding_ratio=padding_ratio, resize_method=resize_method, 
                                                    use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur, sym_aware_training=sym_aware_training,
                                                    sym_input_training=sym_input_training
                                                    )
    print("training_data_folder image example:", rgb_files[obj_id][0], flush=True)

    if training_data_folder_2 != 'none':
        dataset_dir_pbr,_,_,_,_,rgb_files_pbr,_,mask_files_pbr,mask_visib_files_pbr,gts_pbr,gt_infos_pbr,_, camera_params_pbr = bop_io.get_dataset(bop_path, dataset_name, train=True, data_folder=training_data_folder_2, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
        train_dataset_2 = bop_dataset_single_obj_pytorch(
                                                        dataset_dir_pbr, training_data_folder_2, rgb_files_pbr[obj_id], mask_files_pbr[obj_id], mask_visib_files_pbr[obj_id], 
                                                        gts_pbr[obj_id], gt_infos_pbr[obj_id], camera_params_pbr[obj_id], True, 
                                                        BoundingBox_CropSize_image, BoundingBox_CropSize_GT, GT_code_infos, 
                                                        padding_ratio=padding_ratio, resize_method=resize_method,
                                                        use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur, sym_aware_training=sym_aware_training,
                                                        sym_input_training=sym_input_training
                                                    )
        print("training_data_folder_2 image example:", rgb_files_pbr[obj_id][0], flush=True)
        train_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size_2_dataset, shuffle=True, num_workers=num_workers, drop_last=True)                     
        train_loader_2_iter = iter(train_loader_2)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_1_dataset, shuffle=True, num_workers=num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        print("training_data_folder_2 image none")
    # define test data loader
    if not bop_challange:
        dataset_dir_test,_,_,_,_,test_rgb_files,_,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_dataset(bop_path, dataset_name,train=False, data_folder=val_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
        if dataset_name == 'ycbv':
            print("select key frames from ycbv test images")
            key_frame_index = ycbv_select_keyframe(Detection_reaults, test_rgb_files[obj_id])
            test_rgb_files_keyframe = [test_rgb_files[obj_id][i] for i in key_frame_index]
            test_mask_files_keyframe = [test_mask_files[obj_id][i] for i in key_frame_index]
            test_mask_visib_files_keyframe = [test_mask_visib_files[obj_id][i] for i in key_frame_index]
            test_gts_keyframe = [test_gts[obj_id][i] for i in key_frame_index]
            test_gt_infos_keyframe = [test_gt_infos[obj_id][i] for i in key_frame_index]
            camera_params_test_keyframe = [camera_params_test[obj_id][i] for i in key_frame_index]
            test_rgb_files[obj_id] = test_rgb_files_keyframe
            test_mask_files[obj_id] = test_mask_files_keyframe
            test_mask_visib_files[obj_id] = test_mask_visib_files_keyframe
            test_gts[obj_id] = test_gts_keyframe
            test_gt_infos[obj_id] = test_gt_infos_keyframe
            camera_params_test[obj_id] = camera_params_test_keyframe
    else:
        dataset_dir_test,_,_,_,_,test_rgb_files,_,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_bop_challange_test_data(bop_path, dataset_name, target_obj_id=obj_id+1, data_folder=val_folder)
    print('test_rgb_file exsample', test_rgb_files[obj_id][0])

    if Detection_reaults != 'none':
        Det_Bbox = get_detection_results(Detection_reaults, test_rgb_files[obj_id], obj_id+1, 0)
    else:
        Det_Bbox = None

    test_dataset = bop_dataset_single_obj_pytorch(
                                            dataset_dir_test, val_folder, test_rgb_files[obj_id], test_mask_files[obj_id], test_mask_visib_files[obj_id], 
                                            test_gts[obj_id], test_gt_infos[obj_id], camera_params_test[obj_id], False, 
                                            BoundingBox_CropSize_image, BoundingBox_CropSize_GT, GT_code_infos, 
                                            padding_ratio=padding_ratio, resize_method=resize_method, Detect_Bbox=Det_Bbox,
                                            use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur, sym_aware_training=sym_aware_training,
                                            sym_input_training=sym_input_training
                                        )
    
    print("number of test images: ", len(test_dataset), flush=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=num_workers)

    #############build the network 
    binary_code_length = number_of_itration
    print("binary_code_length: ", binary_code_length)
    configs['binary_code_length'] = binary_code_length

    if backbone_name == 'mobilenetv3':
        net = MobileNetV3Seg(
            num_classes=binary_code_length, 
            concat=concat, 
            binary_code_length=binary_code_length, 
            divided_number_each_iteration = divide_number_each_itration, 
            output_kernel_size = output_kernel_size,
            efficientnet_key = efficientnet_key,
            sym_input_training=sym_input_training
        )
    elif backbone_name == 'repvit':
        net = RepViTNet(
                binary_code_length=binary_code_length,
                model_name='repvit_m1_1.dist_300e_in1k',
                pretrained=True,
        )
    else:
        net = BinaryCodeNet_Deeplab(
                    num_resnet_layers=resnet_layer, 
                    concat=concat, 
                    binary_code_length=binary_code_length, 
                    divided_number_each_iteration = divide_number_each_itration, 
                    output_kernel_size = output_kernel_size,
                    efficientnet_key = efficientnet_key,
                    sym_input_training=sym_input_training
                )
        
    maskLoss = MaskLoss()
    if use_sym_aware_loss:
        print("use_sym_aware_loss")
        binarycode_loss = SymLossWeighted(BinaryCode_Loss_Type, mask_binary_code_loss, divide_number_each_itration, use_histgramm_weighted_binary_loss=use_histgramm_weighted_binary_loss, sym_bit=sym_bit)
    else:
        binarycode_loss = BinaryCodeLoss(BinaryCode_Loss_Type, mask_binary_code_loss, divide_number_each_itration, use_histgramm_weighted_binary_loss=use_histgramm_weighted_binary_loss)
    #visulize input image, ground truth code, ground truth mask
    writer = SummaryWriter(tensorboard_path)

    #visulize_input_data_and_network(writer, train_loader, net)
    
    if configs['use_cuda'] and torch.cuda.is_available():
        net=net.cuda()

    if optimizer_type == 'SGD':
        optimizer=optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'Adam':
        optimizer=optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError(f"unknown optimizer type: {optimizer_type}")

    best_score_path = os.path.join(check_point_path, 'best_score')
    if not os.path.isdir(best_score_path):
        os.makedirs(best_score_path)
    best_score = 0
    best_score_5 = 0
    best_score_25 = 0
    
    epoch = 0
    if load_pretrain_checkpoint:
        pretrain_path = os.path.join(pretrain_path, obj_name, obj_name)
        checkpoint = torch.load( pretrain_path )['model_state_dict']
        net = load_partial_weights(net, checkpoint)
        print(f"Successfully loaded ", pretrain_path, "and adjusted ")                
        # net.load_state_dict(checkpoint['model_state_dict'])
        # print('load pretrained model from ', pretrain_path)


    if load_checkpoint:
        try:
            checkpoint = torch.load( get_checkpoint(best_score_path) )
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_score = checkpoint['best_score']
            epoch = checkpoint['epoch']
            print('load checkpoint from ', get_checkpoint(best_score_path), "with best score ", best_score, "at epoch ", epoch)
        except:
            print("no checkpoint found, start training from scratch")
            epoch = 0

    # train the network
    while True:
        epoch += 1
        if epoch > total_epoch:
            print('end the training in epoch:', epoch)
            break        
        # end_training = False
        iteration_step = 0
        loss_sum = 0
        loss_m_sum = 0
        loss_b_sum = 0
        for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(train_loader):
            # if multiple training sets, get data from the second set
            if training_data_folder_2 != 'none':
                try:
                    data_2, entire_masks_2, masks_2, Rs_2, ts_2, Bboxes_2, class_code_images_2, cam_Ks_2 = next(train_loader_2_iter)
                except StopIteration:
                    train_loader_2_iter = iter(train_loader_2)
                    data_2, entire_masks_2, masks_2, Rs_2, ts_2, Bboxes_2, class_code_images_2, cam_Ks_2 = next(train_loader_2_iter)

                data = torch.cat((data, data_2), 0)
                entire_masks = torch.cat((entire_masks, entire_masks_2), 0)
                masks = torch.cat((masks, masks_2), 0)
                class_code_images = torch.cat((class_code_images, class_code_images_2), 0)
                Rs = torch.cat((Rs, Rs_2), 0)
                ts = torch.cat((ts, ts_2), 0)
                cam_Ks = torch.cat((cam_Ks, cam_Ks_2), 0)
                Bboxes = torch.cat((Bboxes, Bboxes_2), 0)
            # data to GPU
            if configs['use_cuda'] and torch.cuda.is_available():
                data=data.cuda()
                entire_masks=entire_masks.cuda()
                masks = masks.cuda()
                class_code_images = class_code_images.cuda()
         
            optimizer.zero_grad()
            if data.shape[0]!= batch_size:
                raise ValueError(f"batch size wrong")
            pred_mask_prob, pred_code_prob = net(data)

            # loss for predicted binary coding
            pred_mask_for_loss = from_output_to_class_mask(pred_mask_prob)
            pred_mask_for_loss = torch.tensor(pred_mask_for_loss)
            if configs['use_cuda'] and torch.cuda.is_available():
                pred_mask_for_loss = pred_mask_for_loss.cuda()

            loss_b = binarycode_loss(pred_code_prob, pred_mask_for_loss, class_code_images)

            # loss for predicted mask
            if predict_entire_mask:
                loss_m = maskLoss(pred_mask_prob, entire_masks)
            else:
                loss_m = maskLoss(pred_mask_prob, masks)

            loss = binary_loss_weight*loss_b + loss_m 
       
            loss.backward()
            optimizer.step()
        
            print(config_file_name, " iteration_step:", iteration_step, 
                "loss_b:", loss_b.item(),  
                "loss_m:", loss_m.item() ,  
                "loss:", loss.item(),
                flush=True
            )
            loss_sum += loss.item()
            loss_m_sum += loss_m.item()
            loss_b_sum += loss_b.item()
            
            # if iteration_step == 0:
            #     ADD_passed = test_network_with_single_obj(net, test_loader, obj_diameter, writer, dict_class_id_3D_points, vertices, epoch, configs, 0,calc_add_and_adi=False)
            #     print("ADD_passed", ADD_passed)
            #     if ADD_passed >= best_score:
            #         best_score = ADD_passed
            #         print("best_score", best_score)
            #         save_best_checkpoint(best_score_path, net, optimizer, best_score, epoch)
            #         save_image(image_path, 0, data, pred_mask_prob, pred_code_prob, class_code_images,BinaryCode_Loss_Type, divide_number_each_itration, binary_code_length)
            iteration_step += 1


        writer.add_scalar('Loss/training loss total', loss_sum/iteration_step, epoch)
        writer.add_scalar('Loss/training loss mask', loss_m_sum/iteration_step, epoch)
        writer.add_scalar('Loss/training loss binary code', loss_b_sum/iteration_step, epoch)

        # # test the trained CNN
        # log_freq = 5

        # if (epoch)%log_freq == 0:
        #     if binarycode_loss.histogram is not None:
        #         np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
        #         print('Train err:{}'.format(binarycode_loss.histogram.detach().cpu().numpy()))
        #     save_checkpoint(check_point_path, net, epoch, best_score, optimizer, 3)

            
        if epoch == 1 or epoch % 10 == 0 or epoch >= 50:
            ADD_passed, ADD_passed_5, ADD_passed_25 = test_network_with_single_obj(net, test_loader, obj_diameter, writer, dict_class_id_3D_points, vertices, epoch, configs, 0,calc_add_and_adi=False)
            print("ADD_passed", ADD_passed)
            if ADD_passed >= best_score:
                best_score = ADD_passed
                print("best_score", best_score)
                save_best_checkpoint(best_score_path, net, optimizer, best_score, epoch)
                save_image(image_path, 0, data, pred_mask_prob, pred_code_prob, class_code_images,BinaryCode_Loss_Type, divide_number_each_itration, binary_code_length)
            # elif ADD_passed_5 >= best_score_5:
            #     best_score_5 = ADD_passed_5
            #     print("best_score", best_score_5)
            #     save_best_checkpoint(best_score_path, net, optimizer, best_score_5, epoch)
            #     save_image(image_path, 0, data, pred_mask_prob, pred_code_prob, class_code_images,BinaryCode_Loss_Type, divide_number_each_itration, binary_code_length)
            # elif ADD_passed_25 >= best_score_25:
            #     best_score_25 = ADD_passed_25
            #     print("best_score", best_score_25)
            #     save_best_checkpoint(best_score_path, net, optimizer, best_score_25, epoch)
            #     save_image(image_path, 0, data, pred_mask_prob, pred_code_prob, class_code_images,BinaryCode_Loss_Type, divide_number_each_itration, binary_code_length)
                        
            



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