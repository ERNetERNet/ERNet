import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import matplotlib
import SimpleITK as sitk
from math import exp
import nibabel as nib
import nibabel.processing
import random



def load_data_no_fix(set_name,batch_size,ifshuffle=True):
    
    cur_path = os.getcwd()
    set_path=cur_path+'/dataset/'+set_name
  
    data=np.load(set_path,allow_pickle=True)
    data_set = MyDataset_no_fix(data)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                          batch_size=batch_size, 
                                          shuffle=ifshuffle)
    return loader

class MyDataset_no_fix(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0],self.data[idx][1],self.data[idx][2]


def create_folder(path,folder_name):
    folder_path=path+'/'+folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return

def create_log(log,path,file_name):
    with open(path+"/"+file_name+".txt", 'w') as output:
        output.write(str(log) + '\n')
    return

def append_log(log,path,file_name):
    with open(path+"/"+file_name+".txt", 'a+') as output:
        output.write(str(log) + '\n')
    return

def get_translation_3D(tx, ty, tz):
    t=torch.zeros(1, 4, 4)
    t[:, :, :4] = torch.tensor([[1, 0, 0, tx],
                                [0, 1, 0, ty],
                                [0, 0, 1, tz],
                                [0, 0, 0, 1]])
    return t
    
def get_rotate_3D_x(angle):
    r_x=torch.zeros(1, 4, 4)
    angle = np.radians(angle)
    r_x[:, :, :4] = torch.tensor([[1, 0, 0, 0],
                                  [0, np.cos(angle), -np.sin(angle), 0],
                                  [0, np.sin(angle),  np.cos(angle), 0],
                                  [0, 0, 0, 1]])
    return r_x

def get_rotate_3D_y(angle):
    r_y=torch.zeros(1, 4, 4)
    angle = np.radians(angle)
    r_y[:, :, :4] = torch.tensor([[np.cos(angle), 0, np.sin(angle), 0],
                                  [0, 1, 0, 0],
                                  [-np.sin(angle), 0,  np.cos(angle), 0],
                                  [0, 0, 0, 1]])
    return r_y

def get_rotate_3D_z(angle):
    r_z=torch.zeros(1, 4, 4)
    angle = np.radians(angle)
    r_z[:, :, :4] = torch.tensor([[np.cos(angle), -np.sin(angle), 0, 0],
                                  [np.sin(angle),  np.cos(angle), 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    return r_z

def get_scale_3D(sx,sy,sz):
    t=torch.zeros(1, 4, 4)
    t[:, :, :4] = torch.tensor([[sx, 0, 0, 0],
                                [0, sy, 0, 0],
                                [0, 0, sz, 0],
                                [0, 0, 0, 1]])
    return t


def random_affine_3D(img_size,degree,voxel,scale_min,scale_max):
    angle_x=random.uniform(-degree,degree)
    angle_y=random.uniform(-degree,degree)
    angle_z=random.uniform(-degree,degree)
    
    tx=random.uniform(-voxel,voxel)    
    ty=random.uniform(-voxel,voxel)    
    tz=random.uniform(-voxel,voxel) 

    sx=random.uniform(scale_min,scale_max)
    sy=random.uniform(scale_min,scale_max)
    sz=random.uniform(scale_min,scale_max)
    
    c=get_translation_3D(-img_size//2, -img_size//2,-img_size//2)
    c_inv=torch.inverse(c)
    
    r_x=get_rotate_3D_x(angle_x)
    r_y=get_rotate_3D_y(angle_y)
    r_z=get_rotate_3D_z(angle_z)
    
    t=get_translation_3D(tx, ty, tz)
    s=get_scale_3D(sx,sy,sz)

    A=c_inv@t@r_x@r_y@r_z@s@c
    return A

def param2theta(param, x,y,z):
    param = np.linalg.inv(param)
    theta = np.zeros([3,4])
    theta[0,0] = param[0,0]
    theta[0,1] = param[0,1]*y/x
    theta[0,2] = param[0,2]*z/x
    theta[0,3] = theta[0,0]+ theta[0,1] +  theta[0,2] + 2*param[0,3]/x -1

    theta[1,0] = param[1,0]*x/y
    theta[1,1] = param[1,1]
    theta[1,2] = param[1,2]*z/y
    theta[1,3] = theta[1,1]+ theta[1,2]+ 2*param[1,3]/y + theta[1,0] -1

    theta[2,0] = param[2,0]*x/z
    theta[2,1] = param[2,1]*y/z
    theta[2,2] = param[2,2]
    theta[2,3] = theta[2,2] + 2*param[2,3]/z + theta[2,0]+theta[2,1] -1

    return theta

def train_aug(img,label,img_size,degree,voxel,scale_min,scale_max):
    A=random_affine_3D(img_size,degree,voxel,scale_min,scale_max)
    param=A.cpu().detach().numpy().reshape(4,4)
    theta=param2theta(param, img_size,img_size,img_size)
    theta_torch=torch.from_numpy(theta)
    theta_torch=theta_torch.view(-1,3,4)
    theta_torch=theta_torch.type(torch.FloatTensor).cuda()
    grid = F.affine_grid(theta_torch, torch.Size([1 ,1, img_size,img_size,img_size]),align_corners=True).cuda()

    img_transformed = F.grid_sample(img, grid,mode='bilinear',align_corners=True,padding_mode="zeros")
    label_transformed = F.grid_sample(img, grid,mode='nearest',align_corners=True,padding_mode="zeros")
    
    return img_transformed,label_transformed

def get_ave_std(data):
    ave=torch.mean(torch.FloatTensor(data))
    std=torch.std(torch.FloatTensor(data))
    return ave,std


def f1_dice_loss(y_true, y_pred, is_training=False):

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def compute_label_dice(gt, pred,dice_label):
    gt=gt.squeeze()
    pred=pred.squeeze()
    if dice_label == "LPBA40":
        cls_lst = [-24320., -24064., -23808., -23552., -23296., -23040., -19200.,
           -18944.,   5376.,   5632.,   5888.,   6144.,   6400.,
             6656.,   6912.,   7168.,   7424.,   7680.,   7936.,   8192.,
             8448.,   8704.,  10496.,  10752.,  11008.,  11264.,  11520.,
            11776.,  12032.,  12288.,  12544.,  12800.,  15616.,  15872.,
            16128.,  16384.,  16640.,  16896.,  17152.,  17408.,  20736.,
            20992.,  21248.,  21504.,  21760.,  22016.,  22272.,  22528.,
            22784.,  23040.,  23296.,  23552.,  25856.,  26112.,  30976.,
            31232.,]
    elif dice_label == "IBSR":
        cls_lst = [2., 3., 4., 5., 7., 8., 10., 11., 12., 13., 14., 15., 
                   16., 17., 18., 26., 28., 41., 42., 43., 44., 46., 47., 49., 
                   50., 51., 52., 53., 54., 58., 60.]
    elif dice_label == "CC359":
        cls_lst = [0.,1.]
    
    dice_lst = []
    for cls in cls_lst:
        dice = DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return torch.mean(torch.FloatTensor(dice_lst))

def save_nii_any(epoch,img_name,img,img_path):
    ref_img_GetOrigin=(0.0, 0.0, 0.0)
    ref_img_GetDirection=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    ref_img_GetSpacing=(1.0, 1.0, 1.0)

    img = sitk.GetImageFromArray(img.squeeze().cpu().detach().numpy())
    
    img.SetOrigin(ref_img_GetOrigin)
    img.SetDirection(ref_img_GetDirection)
    img.SetSpacing(ref_img_GetSpacing)
    
    name_img=img_name+"_"+str(epoch)+".nii.gz"
    
    sitk.WriteImage(img, img_path+"/"+name_img)
    
    return

def save_sample_any(epoch,img_name,img,img_path):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    torch_img=img.squeeze()
    x,y,z=torch_img.shape

    torch_lr=torch_img.permute(0,1,2)
    torch_lr=torch_lr.view(x,1,y,z)

    torch_fb=torch_img.permute(2,0,1)
    torch_fb=torch_fb.view(y,1,x,z)
    torch_fb=torch_fb.permute(0, 1, 2,3).flip(2)

    torch_td=torch_img.permute(1,0,2)
    torch_td=torch_td.view(z,1,x,y)
    torch_td=torch_td.permute(0, 1, 2,3).flip(2)
    
    cat_image=torch.cat((torch_lr[x//2], torch_fb[y//2], torch_td[z//2]))
    cat_image=cat_image.view(3,1,x,y)
    
    name_img=img_name+"_"+str(epoch)+".png"
    image_o=np.transpose(vutils.make_grid(cat_image.to(device), nrow=3, normalize=True).cpu(),(1,2,0)).numpy()
    
    matplotlib.image.imsave(img_path+"/"+name_img,image_o)
    
    return
