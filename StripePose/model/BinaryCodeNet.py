from numpy import mean
import torch.nn as nn
import torch
from model.aspp import ASPP, ASPP_50, ASPP_non_binary_ablationstudy, ASPP_Efficientnet, ASPP_Efficientnet_upsampled,ASPP_Efficientnetv2
from model.resnet import ResNet18_OS8,ResNet50_OS8, ResNet34_OS8
from model.efficientnet import efficientnet_intermediate_out,efficientnet_upsampled
import ast
#################### loss ######################
################################################
class SymLossWeighted(nn.Module):
    def __init__(self, binary_code_loss_type, mask_binary_code_loss, divided_number_each_iteration, use_histgramm_weighted_binary_loss=False, sym_bit=[0,2]):
        super().__init__()            
        self.binary_code_loss_type = binary_code_loss_type
        self.mask_binary_code_loss = mask_binary_code_loss
        self.divided_number_each_iteration = divided_number_each_iteration
        self.use_histgramm_weighted_binary_loss = use_histgramm_weighted_binary_loss
        base_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.sym_loss_module = SymLoss(base_loss)
        self.binary_loss_weighted = BinaryLossWeighted(base_loss)
        self.sym_bit = sym_bit # hard coded for sym bits
        # if self.use_histgramm_weighted_binary_loss: # this Hammloss will be used in both case, for loss, or for histogramm
        #     self.hamming_loss = HammingLoss()

        
    def forward(self, pred_binary_code, pred_mask, groundtruth_code):
        
        # p = 2.5 # 幂指数（越大，后期越陡）
        # bin_weights = torch.tensor([2 - 0.5 * ((i) / 23) ** p for i in range(pred_binary_code.shape[1])]) # 先慢后快的衰减
        # print("bin_weights", bin_weights)
        # alpha = 4
        # bin_weights = torch.tensor([1 + torch.exp(-alpha * torch.tensor(i) / 23) for i in range(pred_binary_code.shape[1])])
        bin_weights = torch.tensor([1 for i in range(pred_binary_code.shape[1])])

        # if self.use_histgramm_weighted_binary_loss:
        #     loss_hamm, histogram_new = self.hamming_loss(pred_binary_code, groundtruth_code, pred_mask.clone().detach())
        #     self.histogram  = histogram_new
        #     ## soft bin weigt decrease 
        #     hist_soft = torch.minimum(self.histogram,0.51-self.histogram)
        #     bin_weights = torch.exp(hist_soft*3).clone()    
        bin_weights = bin_weights.to(pred_binary_code.device)
        if self.mask_binary_code_loss:
            pred_binary_code = pred_mask.clone().detach() * pred_binary_code        
        # split the pred mask into 2 parts: pred_sym and pred_mask

        sym = torch.tensor([True if i in self.sym_bit else False for i in range(pred_binary_code.shape[1])])
        other_sym = ~sym
        loss_sym = self.sym_loss_module(pred_binary_code[:, sym, :, :], groundtruth_code[:, sym, :, :], pred_mask, bin_weights[sym])
        loss_other = self.binary_loss_weighted(pred_binary_code[:, other_sym, :, :], groundtruth_code[:, other_sym, :, :],bin_weights[other_sym])
        loss = loss_sym * torch.sum(bin_weights[sym]) + loss_other * torch.sum(bin_weights[other_sym])
        loss = loss / torch.sum(bin_weights)
        # print("loss_sym:", loss_sym.item(), "loss_other:", loss_other.item())
        # print("bin_weights_sym:", bin_weights[sym], "bin_weights_other:", bin_weights[other_sym])
        # print("total_loss:", loss.item())
        # print("=========================================")
        return loss


class BinaryCodeLoss(nn.Module):
    def __init__(self, binary_code_loss_type, mask_binary_code_loss, divided_number_each_iteration, use_histgramm_weighted_binary_loss=False):
        super().__init__()
        self.binary_code_loss_type = binary_code_loss_type
        self.mask_binary_code_loss = mask_binary_code_loss
        self.divided_number_each_iteration = divided_number_each_iteration
        self.use_histgramm_weighted_binary_loss = use_histgramm_weighted_binary_loss

        if self.use_histgramm_weighted_binary_loss: # this Hammloss will be used in both case, for loss, or for histogramm
            self.hamming_loss = HammingLoss()

        if binary_code_loss_type == "L1":
            self.loss = nn.L1Loss(reduction="mean")
        elif binary_code_loss_type == "BCE":
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        elif binary_code_loss_type == "CE":
            self.loss = nn.CrossEntropyLoss(reduction="mean")
        else:
            raise NotImplementedError(f"unknown mask loss type: {binary_code_loss_type}")

        if self.use_histgramm_weighted_binary_loss:
            assert binary_code_loss_type == "BCE"  # currently only have the implementation with BCEWithLogitsLoss
            self.loss= BinaryLossWeighted(nn.BCEWithLogitsLoss(reduction='none'))

        self.histogram= None
        
    def forward(self, pred_binary_code, pred_mask, groundtruth_code):
        ## calculating hamming loss and bit error histogram for loss weighting
        if self.use_histgramm_weighted_binary_loss:
            loss_hamm, histogram_new = self.hamming_loss(pred_binary_code, groundtruth_code, pred_mask.clone().detach())
            if self.histogram is None:
                self.histogram  = histogram_new
            else:
                self.histogram = histogram_new*0.05+self.histogram*0.95
            
            ## soft bin weigt decrease 
            hist_soft = torch.minimum(self.histogram,0.51-self.histogram)
            bin_weights = torch.exp(hist_soft*3).clone()    

        if self.mask_binary_code_loss:
            pred_binary_code = pred_mask.clone().detach() * pred_binary_code

        if self.binary_code_loss_type == "L1":
            pred_binary_code = pred_binary_code.reshape(-1, 1, pred_binary_code.shape[2], pred_binary_code.shape[3])
            pred_binary_code = torch.sigmoid(pred_binary_code)
            groundtruth_code = groundtruth_code.view(-1, 1, groundtruth_code.shape[2], groundtruth_code.shape[3])
        elif self.binary_code_loss_type == "BCE" and not self.use_histgramm_weighted_binary_loss:
            pred_binary_code = pred_binary_code.reshape(-1, pred_binary_code.shape[2], pred_binary_code.shape[3])
            groundtruth_code = groundtruth_code.view(-1, groundtruth_code.shape[2], groundtruth_code.shape[3])
        elif self.binary_code_loss_type == "CE":
            pred_binary_code = pred_binary_code.reshape(-1, self.divided_number_each_iteration, pred_binary_code.shape[2], pred_binary_code.shape[3])
            groundtruth_code = groundtruth_code.view(-1, groundtruth_code.shape[2], groundtruth_code.shape[3])
            groundtruth_code = groundtruth_code.long()
        
        if self.use_histgramm_weighted_binary_loss:
            loss = self.loss(pred_binary_code, groundtruth_code, bin_weights)
        else:
            loss = self.loss(pred_binary_code, groundtruth_code)
    
        return loss


class BinaryLossWeighted(nn.Module):
    def __init__(self, baseloss):
        #the base loss should have reduction 'none' 
        super().__init__()
        self.base_loss = baseloss

    def forward(self,input,target,weight):
        base_output=self.base_loss(input,target)
        assert base_output.ndim == 4
        output = base_output.mean([0,2,3])
        output = torch.sum(output*weight)/torch.sum(weight)
        return output


class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")
       
    def forward(self, pred_mask, groundtruth_mask): 
        pred_mask = pred_mask[:, 0, :, :]
        pred_mask = torch.sigmoid(pred_mask)
        
        return self.loss(pred_mask, groundtruth_mask)


class HammingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,predicted_code_prob,GT_code,mask):
        assert predicted_code_prob.ndim ==4
        mask_hard  = mask.round().clamp(0,1) # still kept round and clamp for safety
        code1_hard = torch.sigmoid(predicted_code_prob).round().clamp(0,1)
        code2_hard = GT_code.round().clamp(0,1) # still kept round and clamp for safety
        hamm = torch.abs(code1_hard-code2_hard)*mask_hard
        histogram = hamm.sum([0,2,3])/(mask_hard.sum()+1)
        hamm_loss = histogram.mean()
        
        return hamm_loss,histogram.detach()

# class SymLoss(nn.Module):
#     def __init__(self,base_loss):
#         super().__init__()
#         self.loss = base_loss
       
#     def forward(self, pred_sym, groundtruth_mask,mask,weights): 
#         b = pred_sym.shape[0]
#         # pred_sym = torch.sigmoid(pred_sym)
#         # mask = mask.detach()
#         pred_sym_A = mask.detach() * pred_sym
#         pred_sym_B = mask.detach() * (-pred_sym)
#         loss = 0
#         for i in range(b):
#             pred_sym_i_A = pred_sym_A[i, :, :, :]
#             pred_sym_i_B = pred_sym_B[i, :, :, :]
#             # loss = (pred_sym + groundtruth_mask - 1.0).mean() * (pred_sym - groundtruth_mask).mean()
#             loss_A = self.loss(pred_sym_i_A, groundtruth_mask[i])
#             loss_B = self.loss(pred_sym_i_B, groundtruth_mask[i])
#             loss_i = torch.min((loss_A.mean([1,2])* weights).sum(), (loss_B.mean([1,2])* weights).sum())
#             loss += loss_i / weights.sum()
#         return loss/b
class SymLoss(nn.Module):
    def __init__(self,base_loss):
        super().__init__()
        self.loss = base_loss

    def forward(self, pred_sym, groundtruth_code, mask, weights):
        batch_size = pred_sym.shape[0]
        sym_b = pred_sym.shape[1]
        if sym_b == 2:
            pred_sym_A = mask.detach() * pred_sym
            pred_sym_B = mask.detach() * (-pred_sym)
            loss = 0
            for i in range(batch_size):
                pred_sym_i_A = pred_sym_A[i, :, :, :]
                pred_sym_i_B = pred_sym_B[i, :, :, :]
                # loss = (pred_sym + groundtruth_mask - 1.0).mean() * (pred_sym - groundtruth_mask).mean()
                loss_A = self.loss(pred_sym_i_A, groundtruth_code[i])
                loss_B = self.loss(pred_sym_i_B, groundtruth_code[i])
                loss_i = torch.min((loss_A.mean([1,2])* weights).sum(), (loss_B.mean([1,2])* weights).sum())
                loss += loss_i / weights.sum()
            return loss/batch_size
            # groundtruth_code_B = (mask - groundtruth_code.clone())
            # loss = 0

            # for i in range(batch_size):

            #     # print("groundtruth_code", groundtruth_code[0,0,:,:].cpu().detach().numpy().max(), groundtruth_code[0,0,:,:].cpu().detach().numpy().min())
            #     # import cv2
            #     # cv2.imshow("mask",(mask[0,0,:,:].cpu().detach().numpy()) * 255)
            #     # cv2.imshow("groundtruth_code",(groundtruth_code[0,0,:,:].cpu().detach().numpy()) * 255)
            #     # cv2.imshow("groundtruth_code_B",(groundtruth_code_B[0,0,:,:].cpu().detach().numpy()) * 255)
            #     # cv2.waitKey(0)
            #     # cv2.destroyAllWindows()
            #     output_A=self.loss(pred_sym[i], groundtruth_code[i]).mean([1,2])
            #     output_B=self.loss(pred_sym[i], groundtruth_code_B[i]).mean([1,2])
            #     loss_A = torch.sum(output_A*weights)/torch.sum(weights)
            #     loss_B = torch.sum(output_B*weights)/torch.sum(weights)
            #     loss_i = torch.min(loss_A, loss_B)
            #     loss += loss_i
            # return loss
        else:
            NotImplementedError("currently only support batch size 2  for sym loss, but got sym size {}".format(sym_b))
        # elif b == 3:
        #     pred_sym_A = pred_sym.clone() * mask.detach()
        #     groundtruth_code_B = groundtruth_code.clone()
        #     pred_sym_B[:, :2] *= -1
        #     pred_sym_B = pred_sym_B * mask.detach()
        #     groundtruth_code_C = groundtruth_code.clone()
        #     pred_sym_C[:, [0,2]] *= -1
        #     pred_sym_C = pred_sym_C * mask.detach()
        #     groundtruth_code_D = groundtruth_code.clone()
        #     pred_sym_D[:, [1,2]] *= -1
        #     pred_sym_D = pred_sym_D * mask.detach()

        #     output_A=self.loss(pred_sym_A, groundtruth_code)
        #     output_B=self.loss(pred_sym_B, groundtruth_code)
        #     output_C=self.loss(pred_sym_C, groundtruth_code)
        #     output_D=self.loss(pred_sym_D, groundtruth_code)
        #     output_A = output_A.mean([0,2,3])
        #     output_B = output_B.mean([0,2,3])
        #     output_C = output_C.mean([0,2,3])
        #     output_D = output_D.mean([0,2,3])
        #     loss_A = torch.sum(output_A*weights)/torch.sum(weights)
        #     loss_B = torch.sum(output_B*weights)/torch.sum(weights)
        #     loss_C = torch.sum(output_C*weights)/torch.sum(weights)
        #     loss_D = torch.sum(output_D*weights)/torch.sum(weights)
        #     loss = torch.stack([loss_A, loss_B, loss_C, loss_D], dim=0).min(dim=0)[0]
        #     # loss = torch.minimum(torch.minimum(loss_A, loss_B), torch.minimum(loss_C, loss_D))
        #     return loss

        # pred_sym = torch.sigmoid(pred_sym)
        # mask = mask.detach()

#################### model ######################
################################################

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}



class BinaryCodeNet_Deeplab(nn.Module):
    def __init__(
        self, 
        num_resnet_layers, 
        binary_code_length, 
        divided_number_each_iteration,
        concat=False, 
        output_kernel_size = 1, 
        efficientnet_key = None, 
        sym_input_training = False
    ):
        super(BinaryCodeNet_Deeplab, self).__init__()
        self.concat = concat

        # DeepLabV3 and DeepLabV3_non_binary can be merged as 1 model. To ensure the checkpoint works, we keep it unchanged for now.
        if divided_number_each_iteration == 2:
            # hard coded 1 for object mask
            # we assumed for binary case, no CE loss will be used. Otherwise it will be binary_code_length*2 + 1
            self.net = DeepLabV3(num_resnet_layers, binary_code_length + 1, concat=self.concat, output_kernel_size=output_kernel_size, efficientnet_key=efficientnet_key,sym_input_training=sym_input_training)
        else:
            self.net = DeepLabV3_non_binary(num_resnet_layers, binary_code_length=binary_code_length, divided_number_each_iteration=divided_number_each_iteration, concat=self.concat, output_kernel_size=output_kernel_size)

    def forward(self, inputs):
        return self.net(inputs)


class DeepLabV3(nn.Module):
    def __init__(self, num_resnet_layers, num_classes, concat=False, output_kernel_size=1, efficientnet_key=None,sym_input_training=False):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes
        self.concat = concat
        self.num_resnet_layers = num_resnet_layers
        self.sym_input_training = sym_input_training
        self.efficientnet_key = efficientnet_key

        if efficientnet_key == None:
            if num_resnet_layers == 18:
                self.resnet = ResNet34_OS8(18, concat) # NOTE! specify the type of ResNet here
                self.aspp = ASPP(num_classes=self.num_classes, concat=concat, output_kernel_size=output_kernel_size)
            elif num_resnet_layers == 34:
                self.resnet = ResNet34_OS8(34, concat) # NOTE! specify the type of ResNet here
                self.aspp = ASPP(num_classes=self.num_classes, concat=concat, output_kernel_size=output_kernel_size) 
            elif num_resnet_layers == 50:
                self.resnet = ResNet50_OS8(50, concat) # NOTE! specify the type of ResNet here
                self.aspp = ASPP_50(num_classes=self.num_classes, concat=concat, output_kernel_size=output_kernel_size)
            if self.sym_input_training:
                if not self.concat:
                    self.resnet.resnet[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                else:
                    self.resnet.resnet_layer_1[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()

        else:
            if self.efficientnet_key == 'b5':
                self.efficientnet = efficientnet_intermediate_out(version=self.efficientnet_key,concat=concat)
                self.aspp = ASPP_Efficientnet(num_classes=self.num_classes, concat=concat, output_kernel_size=output_kernel_size)
            elif self.efficientnet_key == 'b4':
                self.efficientnet = efficientnet_upsampled(version=self.efficientnet_key,concat=concat)
                self.aspp = ASPP_Efficientnet_upsampled(num_classes=self.num_classes, concat=concat, output_kernel_size=output_kernel_size) 
        

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        if self.efficientnet_key == None:
            if not self.concat:
                feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))
                output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))
            else:
                x_high_feature, x_128, x_64, x_32, x_16 = self.resnet(x)
                output = self.aspp(x_high_feature, x_128, x_64, x_32, x_16)
                # print("x_high_feature.shape", x_high_feature.shape)
                # print("x_128.shape", x_128.shape)
                # print("x_64.shape", x_64.shape)
                # print("x_32.shape", x_32.shape)
                # print("x_16.shape", x_16.shape)
                
        else:
            if self.efficientnet_key  == 'b5':
                l9,l6,l4,l3,l2 = self.efficientnet(x)
                output = self.aspp(l9,l6,l4,l3,l2)
            elif self.efficientnet_key == 'b4':
                l9,l3,l2 = self.efficientnet(x)
                output = self.aspp(l9,l3,l2)

        #output = F.interpolate(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        mask,binary_code = torch.split(output,[1,self.num_classes-1],1)
        return mask, binary_code


class DeepLabV3_non_binary(nn.Module):
    def __init__(self, num_resnet_layers, binary_code_length=16, divided_number_each_iteration=2, concat=False, output_kernel_size=1):
        super(DeepLabV3_non_binary, self).__init__()

        self.concat = concat
        self.resnet = ResNet34_OS8(34, concat) # NOTE! specify the type of ResNet here
        self.aspp = ASPP_non_binary_ablationstudy(code_output_length=binary_code_length, divided_number_each_iteration=divided_number_each_iteration, concat=concat, output_kernel_size=output_kernel_size) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        if not self.concat:
            feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))
            mask, binary_code = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))
        else:
            x_high_feature, x_128, x_64, x_32, x_16 = self.resnet(x)
            mask, binary_code = self.aspp(x_high_feature, x_128, x_64, x_32, x_16)

        #output = F.interpolate(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return mask, binary_code
    
    