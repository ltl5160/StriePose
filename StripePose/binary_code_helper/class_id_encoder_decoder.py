import math
import numpy as np
import torch 


def RGB_image_to_class_id_image(RGB_image):
    # RGB_image was saved as BGR in opencv
    # the input of this function has to be numpy array, due to the bit shift
    RGB_image = RGB_image.astype(int)
    class_id_B = np.left_shift(RGB_image[:,:,0], 16)
    class_id_G = np.left_shift(RGB_image[:,:,1], 8)
    class_id_R = RGB_image[:,:,2]

    class_id_image = class_id_B + class_id_G + class_id_R
    return class_id_image

def class_code_images_to_class_id_image(class_code_images, class_base=2):
    """
        class_code_images: numpy array HWC
    """

    class_id_image = np.zeros((class_code_images.shape[0], class_code_images.shape[1]))

    codes_length = class_code_images.shape[2]
    for i in range(codes_length):
        class_id_image =  class_id_image + class_code_images[:,:,i] * (class_base**(codes_length - 1 - i))

    return class_id_image

def class_code_images_to_class_id_image_torch(class_code_images, class_base=2):
    """
        class_code_images: torch tensor CHW
    """
    
    class_id_image = np.zeros((class_code_images.shape[1], class_code_images.shape[2]))

    codes_length = class_code_images.shape[0]
    for i in range(codes_length):
        class_id_image =  class_id_image + class_code_images[i,:,:] * (class_base**(codes_length - 1 - i))

    return class_id_image

def class_id_image_to_class_code_images(class_id_image, class_base=2, iteration=8, number_of_class=256):
    """
        class_id_image: 2D numpy array
    """
    if class_base**iteration != number_of_class:
        raise ValueError('this combination of base and itration is not possible')

    iteration = int(iteration)
    class_code_images = np.zeros((class_id_image.shape[0], class_id_image.shape[1], iteration))
    class_id_image = class_id_image.astype(int)

    bit_step = math.log2(class_base)
    
    for i in range(iteration):
        shifted_value_1 = np.right_shift(class_id_image, int(bit_step * (iteration - i -1)))
        shifted_value_2 = np.right_shift(class_id_image, int(bit_step * (iteration - i)))

        temp = shifted_value_1 - shifted_value_2 * (2**bit_step)
        class_code_images[:,:,i] = shifted_value_1 - shifted_value_2 * (2**bit_step)

    return class_code_images

def code_to_id(class_code, class_base=2):
    """
        class_code: 1d array
    """
    id = 0

    codes_length = len(class_code)
    for i in range(codes_length):
        id =  id + class_code[i] * (class_base**(codes_length - 1 - i))

    return id

def str_code_to_id(str_class_code, class_base=2):
    """
        str_class_code: string
    """
    id = 0

    codes_length = len(str_class_code)
    for i in range(codes_length):
        id =  id + int(str_class_code[i]) * (class_base**(codes_length - 1 - i))

    return id

# def fake_A_to_class_code_images(fake_A):
#     fake_A = from_output_to_class_mask(fake_A)
#     mask,_ = torch.split(fake_A , [1,fake_A.shape[1]-1],1)
#     class_code_images = fake_A * mask

#     return class_code_images.cpu().numpy()

def class_code_images_to_class_id_image_torch(class_code_images, class_base=2):
    """
        class_code_images: torch tensor BCHW
    """    
    class_id_image = np.zeros((class_code_images.shape[0], 1, class_code_images.shape[2], class_code_images.shape[3]))
    codes_length = class_code_images.shape[1]
    for i in range(codes_length):
        class_code_image_i = class_code_images[:,i,:,:].reshape(class_id_image.shape)
        class_id_image =  class_id_image +  class_code_image_i * (class_base**(codes_length - 1 - i))
        # print(np.max(class_code_image_i))
        # import cv2
        # cv2.imshow("mask",(class_code_image_i * (class_base**(codes_length - 1 - i)))[0,:,:,:].transpose([1,2,0]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return class_id_image

def class_id_image_to_RGB_image(class_id_image, class_base=2,is_BGR=False):
    # RGB_image was saved as BGR in opencv
    # the input of this function has to be numpy array, due to the bit shift
    codenum =  class_base**8

    class_id_image = class_id_image.astype(int)

    class_id_R = class_id_image % codenum
    # print(np.min(class_id_R))    
    class_id_G = np.right_shift(class_id_image, 8) % codenum
    # print(np.max(class_id_G))    
    class_id_B = np.right_shift(class_id_image, 16) % codenum
    # print(np.max(class_id_B))    
    if is_BGR:
        image = np.concatenate((class_id_B,class_id_G,class_id_R),axis=1)        
    else:
        image = np.concatenate((class_id_R,class_id_G,class_id_B),axis=1)
    # import cv2
    # cv2.imshow("class_id_R",(class_id_R[0,:,:,:]/255).transpose([1,2,0]).astype(float))
    # cv2.imshow("class_id_G",(class_id_G[0,:,:,:]/255).transpose([1,2,0]).astype(float))
    # cv2.imshow("class_id_B",(class_id_B[0,:,:,:]).transpose([1,2,0]).astype(float))
    # cv2.imshow("RGB_image",(RGB_image[0,:,:,:]/255).transpose([1,2,0]).astype(float))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image


def code_to_BGR_image(class_code_images):
    # class_code_images = fake_A_to_class_code_images(fake_A)
    class_id_image = class_code_images_to_class_id_image_torch(class_code_images)
    BGR_image = class_id_image_to_RGB_image(class_id_image,is_BGR=True)
    return BGR_image