import torch 
import os
from binary_code_helper.class_id_encoder_decoder import code_to_BGR_image
import numpy as np

def save_checkpoint(path, net, epoch, best_score, optimizer, max_to_keep):
    if not os.path.isdir(path):
        os.makedirs(path)
    saved_ckpt = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    saved_ckpt = [int(i) for i in saved_ckpt]
    saved_ckpt.sort()
    
    num_saved_ckpt = len(saved_ckpt)
    if num_saved_ckpt >= max_to_keep:
        os.remove(os.path.join(path, str(saved_ckpt[0])))

    torch.save(
                {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_score': best_score
                }, 
                os.path.join(path, str(epoch))
            )

def get_checkpoint(path):
    saved_ckpt = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    saved_ckpt = [str(i) for i in saved_ckpt]
    saved_ckpt.sort()
    return os.path.join(path, str(saved_ckpt[-1]))

def save_best_checkpoint(best_score_path, net, optimizer, best_score, epoch):
    # saved_ckpt = [f for f in os.listdir(best_score_path) if os.path.isfile(os.path.join(best_score_path, f))]
    # if saved_ckpt != []:
    #     os.remove(os.path.join(best_score_path, saved_ckpt[0]))

    best_score_file_name = '{:.4f}'.format(best_score)
    best_score_file_name = best_score_file_name.replace('.', '_')
    best_score_file_name = best_score_file_name + 'step'
    best_score_file_name = best_score_file_name + str(epoch)
    torch.save(
        {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_score,
            'epoch': epoch
        }, 
        os.path.join(best_score_path, best_score_file_name)
    )
    print("best check point saved in ", os.path.join(best_score_path, best_score_file_name))

from PIL import Image
from common_ops import from_output_to_class_mask, from_output_to_class_binary_code
import cv2
def save_image(path, epoch, rgb, pred_mask_prob, pred_code_prob, class_code_images, BinaryCode_Loss_Type, divided_num_each_interation=2, binary_code_length=16)  :
    if not os.path.isdir(path):
        os.makedirs(path)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_numpy = rgb.data[0].cpu().float().numpy()  # convert it into a numpy array
    rgb_numpy = (np.transpose(rgb_numpy[:3,:,:], (1, 2, 0)) * std + mean ) * 255.0  # post-processing: tranpose and scaling
    # image_pil = Image.fromarray(rgb_numpy)
    cv2.imwrite(os.path.join(path, 'epoch%.3d_%s.png' % (epoch, 'rgb')), rgb_numpy)

    pred_masks = from_output_to_class_mask(pred_mask_prob) 
    pred_codes = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, divided_num_each_interation=divided_num_each_interation, binary_code_length=binary_code_length)
    pred_codes = pred_codes * pred_masks
    pred_codes = pred_codes
    pred_zebra_image = code_to_BGR_image(pred_codes)    
    pred_zebra_image = pred_zebra_image[0].transpose((1,2,0))
    # pred_zebra_image = cv2.cvtColor(pred_zebra_image,cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path, 'epoch%.3d_%s.png' % (epoch, 'pred_zebra')), pred_zebra_image)    

    pred_masks = pred_masks[0] * 255
    pred_masks = np.tile(pred_masks, (3, 1, 1)).transpose((1,2,0))
    cv2.imwrite(os.path.join(path, 'epoch%.3d_%s.png' % (epoch, 'pred_mask')) , pred_masks)   

    
    codes = class_code_images.cpu().numpy()
    zebra_image = code_to_BGR_image(codes)
    zebra_image = zebra_image[0].transpose((1,2,0))
    # zebra_image = cv2.cvtColor(zebra_image,cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path, 'epoch%.3d_%s.png' % (epoch, 'real_zebra')),zebra_image)   