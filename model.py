import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import matplotlib
from torch.nn import Parameter
import math
import random


class conv_block(nn.Module):
    def __init__(self, inChan, outChan, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv3d(inChan, outChan, kernel_size=3, stride=stride, padding=1, bias=True),
                nn.BatchNorm3d(outChan),
                nn.LeakyReLU(0.2, inplace=True)
                )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        x = self.conv(x)
        return x

    
class Unet(nn.Module):
    def __init__(self, enc_nf=[2,16,32,32,64,64], dec_nf=[64,32,32,32,16,3]):
        super(Unet, self).__init__()

        self.inconv = conv_block(enc_nf[0], enc_nf[1])
        self.down1 = conv_block(enc_nf[1], enc_nf[2], 2)
        self.down2 = conv_block(enc_nf[2], enc_nf[3], 2)
        self.down3 = conv_block(enc_nf[3], enc_nf[4], 2)
        self.down4 = conv_block(enc_nf[4], enc_nf[5], 2)
        self.up1 = conv_block(enc_nf[-1], dec_nf[0])
        self.up2 = conv_block(dec_nf[0]+enc_nf[4], dec_nf[1])
        self.up3 = conv_block(dec_nf[1]+enc_nf[3], dec_nf[2])
        self.up4 = conv_block(dec_nf[2]+enc_nf[2], dec_nf[3])
        self.same_conv = conv_block(dec_nf[3]+enc_nf[1], dec_nf[4])
        self.outconv = nn.Conv3d(
                dec_nf[4], dec_nf[5], kernel_size=3, stride=1, padding=1, bias=True)
        #init last_conv
        self.outconv.weight.data.normal_(mean=0, std=1e-5)
        if self.outconv.bias is not None:
            self.outconv.bias.data.zero_()

    def forward(self, x):
        # down-sample path (encoder)
        skip1 = self.inconv(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        x = self.down4(skip4)
        # up-sample path (decoder)
        x = self.up1(x)
#        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip4), 1)
        x = self.up2(x)
#        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip3), 1)
        x = self.up3(x)
#        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip2), 1)
        x = self.up4(x)
#        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip1), 1)
        x = self.same_conv(x)
        x = self.outconv(x)
        
        return x


class multi_stage_ext(nn.Module):
    def __init__(self, img_size, stage , gamma,
                 enc_nf=[1,16,32,32,64,64], dec_nf=[64,32,32,32,16,1],):
        super(multi_stage_ext, self).__init__()
        self.unet = Unet(enc_nf, dec_nf)
        self.stage=stage
        self.gamma=gamma
    def forward(self,mov,if_train):
        img_size=mov.shape[-1]
        batch_size=mov.shape[0]
        
        striped_list=[]
        mask_list=[]
        
        for i in range(self.stage):
            mask=self.unet(mov)
            mask=torch.nn.Sigmoid()(self.gamma*mask)
            
            if if_train==False:
                mask[mask<0.5]=0.0
                mask[mask>=0.5]=1.0
            
            mov=mask*mov
            
            striped_list.append(mov)
            mask_list.append(mask)
            
        return striped_list,mask_list

def get_batch_identity_theta_4_4(batch_size):
    theta=Variable(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.float)).cuda()
    i_theta=theta.view(4,4).unsqueeze(0).repeat(batch_size,1,1)
    return i_theta

def get_batch_identity_theta_3_4(batch_size):
    theta=Variable(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float)).cuda()
    i_theta=theta.view(3,4).unsqueeze(0).repeat(batch_size,1,1)
    return i_theta

def theta_3_4_to_4_4(batch_size,theta):
    i_theta=get_batch_identity_theta_4_4(batch_size)
    i_theta[:,0:3,:]=theta
    return i_theta
    
def theta_dot(batch_size,theta_cur,theta_pre):
    theta_cur_4_4=theta_3_4_to_4_4(batch_size,theta_cur)
    theta_pre_4_4=theta_3_4_to_4_4(batch_size,theta_pre)
    
    theta_cat_4_4=theta_cur_4_4@theta_pre_4_4
    theta_cat_3_4=theta_cat_4_4[:,0:3,:]
    return theta_cat_3_4     


class encoder(nn.Module):
    def __init__(self, enc_nf=[2,16,32,64,128,256,512]):
        super(encoder, self).__init__()
        
        self.inconv = conv_block(enc_nf[0], enc_nf[1])
        self.down1 = conv_block(enc_nf[1], enc_nf[2], 2)
        self.down2 = conv_block(enc_nf[2], enc_nf[3], 2)
        self.down3 = conv_block(enc_nf[3], enc_nf[4], 2)
        self.down4 = conv_block(enc_nf[4], enc_nf[5], 2)
        self.down5 = conv_block(enc_nf[5], enc_nf[6], 2)
        
        self.fc_loc = nn.Sequential(
            nn.Linear(512*3*3*3, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, 4 * 3))
        
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0], dtype=torch.float))
        
    def forward(self, x):
        
        x = self.inconv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)    
        x = x.view(-1,)
        
        x=self.fc_loc(x)
        
        theta = x.view(-1, 3, 4)

        return theta   
 

class multi_stage_reg(nn.Module):
    def __init__(self, img_size, stage,
                 enc_affine=[2,16,32,64,128,256,512]):
        super(multi_stage_reg, self).__init__()
        self.affine = encoder(enc_affine)
        self.stage=stage

    def forward(self, ref, mov):
        img_size=ref.shape[-1]
        batch_size=ref.shape[0]
        
        warped_list=[]
        grid_list=[]

        theta_previous= get_batch_identity_theta_3_4(batch_size)
        cur_mov=mov
        
        for i in range(self.stage):
            image = torch.cat((ref, cur_mov), 1)
            theta_cur = self.affine(image)
            
            theta_out=theta_dot(batch_size,theta_cur,theta_previous)
            
            cur_grid = F.affine_grid(theta_out, ref.size(),align_corners=True)
            cur_mov = F.grid_sample(mov, cur_grid,mode="bilinear",align_corners=True)
            
            theta_previous=theta_out
            
            warped_list.append(cur_mov)
            grid_list.append(cur_grid)
                   
        return warped_list,grid_list

    
class ERNet(nn.Module):
    def __init__(self, img_size, ext_stage , reg_stage, gamma):
        
        super(ERNet, self).__init__()
        
        self.ext_net = multi_stage_ext(img_size, ext_stage , gamma)
        self.reg_net = multi_stage_reg(img_size,reg_stage)
    
    def forward(self,ref, mov,if_train):
        striped_list,mask_list = self.ext_net(mov,if_train)
        warped_list,grid_list = self.reg_net(ref,striped_list[-1])
        
        return striped_list,mask_list,warped_list,grid_list

       
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        win = [9] * ndims if self.win is None else self.win

        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)    


class GCC(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(GCC, self).__init__()
 
    def forward(self, I, J):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J

        I_ave, J_ave= I.mean(), J.mean()
        I2_ave, J2_ave = I2.mean(), J2.mean()
        IJ_ave = IJ.mean()
        
        cross = IJ_ave - I_ave * J_ave
        I_var = I2_ave - I_ave.pow(2)
        J_var = J2_ave - J_ave.pow(2)
 
#        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)#1e-5
        cc = cross / (I_var.sqrt() * J_var.sqrt() + np.finfo(float).eps)#1e-5
 
        return -1.0 * cc + 1    
    
    
class first_Grad(nn.Module):
    """
    N-D gradient loss
    """
    def __init__(self, penalty):
        super(first_Grad, self).__init__()
        self.penalty = penalty
    
    def forward(self, pred):
        
        dy = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]) 
        dx = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]) 
        dz = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]) 
        
        if self.penalty == 'l2':
            
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        elif self.penalty == 'l1':
            
            dy = dy
            dx = dx
            dz = dz
        
        
        d = torch.mean(dy) + torch.mean(dx) + torch.mean(dz)
        grad = d / 3.0
        
        return grad  