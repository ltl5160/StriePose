from calendar import c
import torch
import numpy as np

from common_ops import from_output_to_class_mask, from_output_to_class_binary_code,calc_prediction_confidence_image
from tools_for_BOP.common_dataset_info import get_obj_info

from binary_code_helper.CNN_output_to_pose import CNN_outputs_to_object_pose

from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP

from tqdm import tqdm

from bop_toolkit_lib import pose_error
from bop_toolkit_lib import misc

def test_network_with_single_obj(
        net, dataloader, obj_diameter, writer, dict_class_id_3D_points, vertices, step, configs, ignore_n_bit=0,calc_add_and_adi=True,dataname='test'):
    
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']
    binary_code_length = configs['binary_code_length']
    divide_number_each_itration = int(configs['divide_number_each_itration'])
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']
    obj_name = configs['obj_name']
    dataset_name=configs['dataset_name']
    # if 'use_confidence' in configs.keys():
    #     use_confidence = configs['use_confidence']
    #     confidence_threshold = configs['confidence_threshold']
    # else:
    #     use_confidence = False
    #     confidence_threshold = 0
    torch.manual_seed(0)      # the both are only good for ablation study
    np.random.seed(0)         # if can be removed in the final experiments
    
    _, symmetry_obj = get_obj_info(dataset_name)
    
    if obj_name in symmetry_obj:
        Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADD_Error_BOP
        main_metric_name = 'ADI'
        supp_metric_name = 'ADD'

    else:
        Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADI_Error_BOP
        main_metric_name = 'ADD'
        supp_metric_name = 'ADI'
    #net to eval model
    net.eval()

    #test with test data
    ADX_passed=0
    ADX_passed_5=0
    ADX_passed_25=0
    ADX_error=0
    AUC_ADX_error=0
    ADY_passed=0
    ADY_error=0
    AUC_ADY_error=0

    n = 0
    print("test dataset", flush=True)
    with torch.no_grad():
        for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(tqdm(dataloader)):
            # do the prediction and get the predicted binary code
            if configs['use_cuda'] and torch.cuda.is_available():
                data=data.cuda()
                masks = masks.cuda()
                entire_masks = entire_masks.cuda()
                class_code_images = class_code_images.cuda()

            pred_masks_prob, pred_code_prob = net(data)
            confidence_image = np.ones((pred_code_prob.shape[0], pred_code_prob.shape[2], pred_code_prob.shape[3]), dtype=np.float32)
            confidence_threshold = 0
            # if use_confidence and (BinaryCode_Loss_Type == "BCE" or BinaryCode_Loss_Type == "L1"): 
            #     confidence_image = calc_prediction_confidence_image(pred_code_prob)
        
            pred_codes = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, divided_num_each_interation=divide_number_each_itration, binary_code_length=binary_code_length)
            pred_masks = from_output_to_class_mask(pred_masks_prob)

            # from binary code to pose
            pred_codes = pred_codes.transpose(0, 2, 3, 1)

            pred_masks = pred_masks.transpose(0, 2, 3, 1)
            pred_masks = pred_masks.squeeze(axis=-1)

            Rs = Rs.detach().cpu().numpy()
            ts = ts.detach().cpu().numpy()
            Bboxes = Bboxes.detach().cpu().numpy()
            cam_Ks= cam_Ks.detach().cpu().numpy()
            
            for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
                if ignore_n_bit!=0 and configs['eval_with_ignore_bits']:
                    R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_codes[counter][:,:,:-ignore_n_bit],confidence_image[counter], confidence_threshold,
                                                                                Bbox, BoundingBox_CropSize_GT, divide_number_each_itration, dict_class_id_3D_points, 
                                                                                intrinsic_matrix=cam_K)
                else:
                    R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_codes[counter], confidence_image[counter], confidence_threshold,
                                                                                Bbox, BoundingBox_CropSize_GT, divide_number_each_itration, dict_class_id_3D_points, 
                                                                                intrinsic_matrix=cam_K)


                n += 1
                adx_error = 10000
                if success:
                    adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
                    if np.isnan(adx_error):
                        adx_error = 10000
                if adx_error < obj_diameter*0.1:
                    ADX_passed += 1
                if adx_error < obj_diameter*0.05:
                    ADX_passed_5 += 1
                if adx_error < obj_diameter*0.025:
                    ADX_passed_25 += 1
                # else:
                #     import cv2
                #     cv2.imshow("mask_",pred_masks[counter])
                #     cv2.imshow("1_",pred_codes[counter,:,:,1])
                #     cv2.imshow("2_",pred_codes[counter,:,:,2])
                #     cv2.imshow("3_",pred_codes[counter,:,:,3])
                #     cv2.imshow("4_",pred_codes[counter,:,:,4])
                #     cv2.imshow("5_",pred_codes[counter,:,:,5])
                #     cv2.imshow("6_",pred_codes[counter,:,:,6])
                #     cv2.imshow("7_",pred_codes[counter,:,:,7])
                #     cv2.imshow("8_",pred_codes[counter,:,:,8])
                #     cv2.imshow("9_",pred_codes[counter,:,:,9])
                #     cv2.imshow("10_",pred_codes[counter,:,:,10])
                #     cv2.imshow("11_",pred_codes[counter,:,:,11])
                #     cv2.imshow("12_",pred_codes[counter,:,:,12])
                #     cv2.imshow("13_",pred_codes[counter,:,:,13])
                #     cv2.imshow("14_",pred_codes[counter,:,:,14])
                #     cv2.imshow("15_",pred_codes[counter,:,:,15])
                #     cv2.imshow("0_",pred_codes[counter,:,:,0])
                #     cv2.imshow("1",class_code_images[counter,:,:,1])
                #     cv2.imshow("2",class_code_images[counter,:,:,2])
                #     cv2.imshow("3",class_code_images[counter,:,:,3])
                #     cv2.imshow("4",class_code_images[counter,:,:,4])
                #     cv2.imshow("5",class_code_images[counter,:,:,5])
                #     cv2.imshow("6",class_code_images[counter,:,:,6])
                #     cv2.imshow("7",class_code_images[counter,:,:,7])
                #     cv2.imshow("8",class_code_images[counter,:,:,8])
                #     cv2.imshow("9",class_code_images[counter,:,:,9])
                #     cv2.imshow("10",class_code_images[counter,:,:,10])
                #     cv2.imshow("11",class_code_images[counter,:,:,11])
                #     cv2.imshow("12",class_code_images[counter,:,:,12])
                #     cv2.imshow("13",class_code_images[counter,:,:,13])
                #     cv2.imshow("14",class_code_images[counter,:,:,14])
                #     cv2.imshow("15",class_code_images[counter,:,:,15])
                #     cv2.imshow("0",class_code_images[counter,:,:,0])
                #     cv2.waitKey(0)
                # cv2.destroyAllWindows()


                ADX_error += adx_error
                AUC_ADX_error += min(100,max(100-adx_error, 0))

                ady_error = 10000
                if success:
                    ady_error = Calculate_Pose_Error_Supp(r_GT, t_GT, R_predict, t_predict, vertices)
                    if np.isnan(ady_error):
                        ady_error = 10000
                if ady_error < obj_diameter*0.1:
                    ADY_passed += 1
                ADY_error += ady_error
                AUC_ADY_error += min(100,max(100-ady_error, 0))



    ADX_passed = ADX_passed/n
    ADX_passed_5 = ADX_passed_5/n
    ADX_passed_25 = ADX_passed_25/n
    ADX_error= ADX_error/n
    AUC_ADX_error = AUC_ADX_error/n
    if writer != 'none':
        writer.add_scalar('{}data/{}_test'.format(dataname,main_metric_name), ADX_passed, step)
        writer.add_scalar('{}data/{}_test_0.05'.format(dataname,main_metric_name), ADX_passed_5, step)
        writer.add_scalar('{}data/{}_test_0.25'.format(dataname,main_metric_name), ADX_passed_25, step)

        # writer.add_scalar('{}data/{}_Error_test'.format(dataname,main_metric_name), ADX_error, step)
        # writer.add_scalar('{}data/AUC_{}_Error_test'.format(dataname,main_metric_name), AUC_ADX_error, step)

    print('{}data_{}/{}_test:'.format(dataname,main_metric_name,main_metric_name), ADX_passed)
    print('{}data_{}/{}_Error_test'.format(dataname,main_metric_name,main_metric_name), ADX_error)
    print('{}data_AUC_{}/AUC_{}_Error_test'.format(dataname,main_metric_name,main_metric_name), AUC_ADX_error)

    # ADY_passed = ADY_passed/n
    # ADY_error= ADY_error/n
    # AUC_ADY_error = AUC_ADY_error/n
    # if writer != 'none':
    #     writer.add_scalar('{}data/{}_test'.format(dataname,supp_metric_name), ADY_passed, step)
    #     writer.add_scalar('{}data/{}_Error_test'.format(dataname,supp_metric_name), ADY_error, step)
    #     writer.add_scalar('{}data/AUC_{}_Error_test'.format(dataname,supp_metric_name), AUC_ADY_error, step)
    # print('{}data_{}/{}_test'.format(dataname,supp_metric_name,supp_metric_name), ADY_passed)
    # print('{}data_{}/{}_Error_test'.format(dataname,supp_metric_name,supp_metric_name), ADY_error)
    # print('{}data_AUC_{}/AUC_{}_Error_test'.format(dataname,supp_metric_name,supp_metric_name), AUC_ADY_error)
    #net back to train mode
    net.train()
    torch.cuda.empty_cache()
    return ADX_passed, ADX_passed_5, ADX_passed_25



# def test_network_with_single_obj(
#         net, dataloader, obj_diameter, writer, dict_class_id_3D_points, vertices, step, configs, ignore_n_bit=0,calc_add_and_adi=True):
    
#     BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']
#     binary_code_length = configs['binary_code_length']
#     divide_number_each_itration = int(configs['divide_number_each_itration'])
#     BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']
#     obj_name = configs['obj_name']
#     dataset_name=configs['dataset_name']
    
#     _, symmetry_obj = get_obj_info(dataset_name)
    
#     if obj_name in symmetry_obj:
#         Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
#         Calculate_Pose_Error_Supp = Calculate_ADD_Error_BOP
#         main_metric_name = 'ADI'
#         supp_metric_name = 'ADD'
#     else:
#         Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
#         Calculate_Pose_Error_Supp = Calculate_ADI_Error_BOP
#         main_metric_name = 'ADD'
#         supp_metric_name = 'ADI'
#     #net to eval model
#     net.eval()

#     #test with test data
#     ADX_passed=np.zeros(len(dataloader.dataset))
#     ADX_error=np.zeros(len(dataloader.dataset))
#     AUC_ADX_error=np.zeros(len(dataloader.dataset))
#     if calc_add_and_adi:
#         ADY_passed=np.zeros(len(dataloader.dataset))
#         ADY_error=np.zeros(len(dataloader.dataset))
#         AUC_ADY_error=np.zeros(len(dataloader.dataset))

#     print("test dataset", flush=True)
#     for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(tqdm(dataloader)):
#         # do the prediction and get the predicted binary code
#         if configs['use_cuda'] and torch.cuda.is_available():
#             data=data.cuda()
#             masks = masks.cuda()
#             entire_masks = entire_masks.cuda()
#             class_code_images = class_code_images.cuda()

#         pred_masks_prob, pred_code_prob = net(data)

#         pred_codes = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, divided_num_each_interation=divide_number_each_itration, binary_code_length=binary_code_length)
#         pred_masks = from_output_to_class_mask(pred_masks_prob)

#         # from binary code to pose
#         pred_codes = pred_codes.transpose(0, 2, 3, 1)

#         pred_masks = pred_masks.transpose(0, 2, 3, 1)
#         pred_masks = pred_masks.squeeze(axis=-1)

#         Rs = Rs.detach().cpu().numpy()
#         ts = ts.detach().cpu().numpy()
#         Bboxes = Bboxes.detach().cpu().numpy()
#         cam_Ks= cam_Ks.detach().cpu().numpy()
        
#         for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
#             if ignore_n_bit!=0 and configs['eval_with_ignore_bits']:
#                 R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_codes[counter][:,:,:-ignore_n_bit],
#                                                                             Bbox, BoundingBox_CropSize_GT, divide_number_each_itration, dict_class_id_3D_points, 
#                                                                             intrinsic_matrix=cam_K)
#             else:
#                 R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_codes[counter], 
#                                                                             Bbox, BoundingBox_CropSize_GT, divide_number_each_itration, dict_class_id_3D_points, 
#                                                                             intrinsic_matrix=cam_K)


#             batchsize = dataloader.batch_size
#             sample_idx = batch_idx * batchsize + counter
            
#             adx_error = 10000
#             if success:
#                 adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
#                 if np.isnan(adx_error):
#                     adx_error = 10000
#             if adx_error < obj_diameter*0.1:
#                 ADX_passed[sample_idx] = 1
#             ADX_error[sample_idx] = adx_error
#             AUC_ADX_error[counter] = min(100,max(100-adx_error, 0))
#             if calc_add_and_adi:
#                 ady_error = 10000
#                 if success:
#                     ady_error = Calculate_Pose_Error_Supp(r_GT, t_GT, R_predict, t_predict, vertices)
#                     if np.isnan(ady_error):
#                         ady_error = 10000
#                 if ady_error < obj_diameter*0.1:
#                     ADY_passed[sample_idx] = 1
#                 ADY_error[sample_idx] = ady_error
#                 AUC_ADY_error[counter] = min(100,max(100-ady_error, 0))
    
#     ADX_passed = np.mean(ADX_passed)
#     ADX_error= np.mean(ADX_error)
#     AUC_ADX_error = np.mean(AUC_ADX_error)
#     writer.add_scalar('TESTDATA_{}/{}_test'.format(main_metric_name,main_metric_name), ADX_passed, step)
#     writer.add_scalar('TESTDATA_{}/{}_Error_test'.format(main_metric_name,main_metric_name), ADX_error, step)
#     writer.add_scalar('TESTDATA_AUC_{}/AUC_{}_Error_test'.format(main_metric_name,main_metric_name), AUC_ADX_error, step)
#     if calc_add_and_adi:
#         ADY_passed = np.mean(ADY_passed)
#         ADY_error= np.mean(ADY_error)
#         AUC_ADY_error = np.mean(AUC_ADY_error)
#         writer.add_scalar('TESTDATA_{}/{}_test'.format(supp_metric_name,supp_metric_name), ADY_passed, step)
#         writer.add_scalar('TESTDATA_{}/{}_Error_test'.format(supp_metric_name,supp_metric_name), ADY_error, step)
#         writer.add_scalar('TESTDATA_AUC_{}/AUC_{}_Error_test'.format(supp_metric_name,supp_metric_name), AUC_ADY_error, step)

#     #net back to train mode
#     net.train()

#     return ADX_passed