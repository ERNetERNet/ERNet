import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from skimage import feature
from skimage import transform
import os
import SimpleITK as sitk
import nibabel.processing


def LPBA40_img2nii():
    for i in range(1,41):
        idx=str(i).rjust(2,'0')

        raw_name='S'+str(idx)+'/'+'S'+str(idx)+'.native.mri.img.gz'        
        bm_label_name='S'+str(idx)+'/'+'S'+str(idx)+'.native.brain.mask.img.gz'
        am_label_name='S'+str(idx)+'.delineation.mri.img'
        skulled_name='S'+str(idx)+'/'+'S'+str(idx)+'.native.brain.bfc.img.gz' 

        raw_name_nii='S'+str(idx)+'/'+'S'+str(idx)+'.native.mri.nii.gz'        
        bm_label_name_nii='S'+str(idx)+'/'+'S'+str(idx)+'.native.brain.mask.nii.gz'
        am_label_name_nii='S'+str(idx)+'.delineation.mri.nii'
        skulled_name_nii='S'+str(idx)+'/'+'S'+str(idx)+'.native.brain.bfc.nii.gz' 

        raw_img=sitk.ReadImage('./dataset/LPBA40subjects.native_space/LPBA40/native_space/'+raw_name)
        bm_label_img=sitk.ReadImage('./dataset/LPBA40subjects.native_space/LPBA40/native_space/'+bm_label_name)
        am_label_img=sitk.ReadImage('./dataset/LPBA40_seg_label_all/'+am_label_name)
        skulled_img=sitk.ReadImage('./dataset/LPBA40subjects.native_space/LPBA40/native_space/'+skulled_name)

        sitk.WriteImage(raw_img,'./dataset/LPBA40subjects.native_space/LPBA40/native_space/'+raw_name_nii)
        sitk.WriteImage(bm_label_img,'./dataset/LPBA40subjects.native_space/LPBA40/native_space/'+bm_label_name_nii)
        sitk.WriteImage(am_label_img,'./dataset/LPBA40_seg_label_all/'+am_label_name_nii)
        sitk.WriteImage(skulled_img,'./dataset/LPBA40subjects.native_space/LPBA40/native_space/'+skulled_name_nii)
        
    return



def image_to_square_v2(img,size):
    img_new=np.zeros((size,size,size))
    z,x,y=img.shape
    if z<= size:
        az = (size-z)//2
        img_new_start_z=az
        img_new_end_z=az+z
        img_start_z=0
        img_end_z=z

    elif z> size:
        az = (z-size)//2
        img_new_start_z=0
        img_new_end_z=size
        img_start_z=az
        img_end_z=az+size

    if x<= size:
        ax = (size-x)//2
        img_new_start_x=ax
        img_new_end_x=ax+x
        img_start_x=0
        img_end_x=x

    elif x> size:
        ax = (x-size)//2
        img_new_start_x=0
        img_new_end_x=size
        img_start_x=ax
        img_end_x=ax+size

    if y<= size:
        ay = (size-y)//2
        img_new_start_y=ay
        img_new_end_y=ay+y
        img_start_y=0
        img_end_y=y

    elif y> size:
        ay = (y-size)//2
        img_new_start_y=0
        img_new_end_y=size
        img_start_y=ay
        img_end_y=ay+size

    img_new[img_new_start_z:img_new_end_z,img_new_start_x:img_new_end_x,img_new_start_y:img_new_end_y] = img[img_start_z:img_end_z,img_start_x:img_end_x,img_start_y:img_end_y]   
    return img_new


def preprocess_LPBA40_96():
    for i in range(1,41):
        idx=str(i).rjust(2,'0')

        raw_name_nii='S'+str(idx)+'/'+'S'+str(idx)+'.native.mri.nii.gz'        
        bm_label_name_nii='S'+str(idx)+'/'+'S'+str(idx)+'.native.brain.mask.nii.gz'
        am_label_name_nii='S'+str(idx)+'.delineation.mri.nii'
        skulled_name_nii='S'+str(idx)+'/'+'S'+str(idx)+'.native.brain.bfc.nii.gz' 

        raw_img=nibabel.load('./dataset/LPBA40subjects.native_space/LPBA40/native_space/'+raw_name_nii)
        bm_label_img=nibabel.load('./dataset/LPBA40subjects.native_space/LPBA40/native_space/'+bm_label_name_nii)
        am_label_img=nibabel.load('./dataset/LPBA40_seg_label_all/'+am_label_name_nii)
        skulled_img=nibabel.load('./dataset/LPBA40subjects.native_space/LPBA40/native_space/'+skulled_name_nii)
        
        voxel_size = [2.31, 2.31, 2.31]
        resampled_raw_img = nibabel.processing.resample_to_output(raw_img, voxel_size,order=1)
        resampled_bm_label_img = nibabel.processing.resample_to_output(bm_label_img, voxel_size,order=0)
        resampled_am_label_img = nibabel.processing.resample_to_output(am_label_img, voxel_size,order=0)
        resampled_skulled_img = nibabel.processing.resample_to_output(skulled_img, voxel_size,order=1)
        
        resampled_raw_np=image_to_square_v2(resampled_raw_img.get_fdata(),96)
        resampled_bm_label_np=image_to_square_v2(resampled_bm_label_img.get_fdata(),96)
        resampled_am_label_np=image_to_square_v2(resampled_am_label_img.get_fdata(),96)
        resampled_skulled_np=image_to_square_v2(resampled_skulled_img.get_fdata(),96)
        
        
        clipped_img = nib.Nifti1Image(resampled_raw_np, resampled_raw_img.affine, nib.Nifti1Header())
        clipped_bm = nib.Nifti1Image(resampled_bm_label_np, resampled_bm_label_img.affine, nib.Nifti1Header())
        clipped_am = nib.Nifti1Image(resampled_am_label_np, resampled_am_label_img.affine, nib.Nifti1Header())
        clipped_skulled = nib.Nifti1Image(resampled_skulled_np, resampled_skulled_img.affine, nib.Nifti1Header())
    
        img_path='./dataset/LPBA40_96/'

        raw_name_save = 'S'+str(idx)+"_"+str(0)+"raw"+"_"+str(96)+".nii.gz"
        bm_label_name_save = 'S'+str(idx)+"_"+str(0)+"bm_label"+"_"+str(96)+".nii.gz"
        am_label_name_save = 'S'+str(idx)+"_"+str(0)+"am_label"+"_"+str(96)+".nii.gz"
        skulled_name_save = 'S'+str(idx)+"_"+str(0)+"skulled"+"_"+str(96)+".nii.gz"

        nib.save(clipped_img, img_path+raw_name_save)
        nib.save(clipped_bm, img_path+bm_label_name_save)
        nib.save(clipped_am, img_path+am_label_name_save)
        nib.save(clipped_skulled, img_path+skulled_name_save)
           
    return


def get_dataset_LPBA40_96(size):
    train_set=[]
    test_set=[]
    val_set=[]

    img_path='./dataset/LPBA40_96/'
    fixed_name='S01_0skulled'+"_"+str(size)+'.nii.gz'
    fixed_img=nibabel.load(img_path+fixed_name)
    fixed_np=norm_to_0_1(fixed_img.get_fdata())
    
    fixed_name_amlabel='S01_0am_label'+"_"+str(size)+'.nii.gz'
    fixed_amlabel=nibabel.load(img_path+fixed_name_amlabel)
    fixed_np_amlabel=fixed_amlabel.get_fdata()
    
    for i in range(11,41):
        idx_train=str(i).rjust(2,'0')

        train_raw_name='S'+idx_train+"_"+str(0)+'raw'+'_'+str(size)+'.nii.gz'
        train_bm_label_name='S'+idx_train+"_"+str(0)+'bm_label'+'_'+str(size)+'.nii.gz'
        train_am_label_name='S'+idx_train+"_"+str(0)+'am_label'+'_'+str(size)+'.nii.gz'

        train_raw=nibabel.load(img_path+train_raw_name)
        train_bm_label=nibabel.load(img_path+train_bm_label_name)
        train_am_label=nibabel.load(img_path+train_am_label_name)

        train_raw_np=norm_to_0_1(train_raw.get_fdata())
        train_bm_label_np=norm_to_0_1(train_bm_label.get_fdata())
        train_am_label_np=train_am_label.get_fdata()

        '''
        train_set.append((fixed_np,train_raw_np,
                          fixed_np_amlabel,
                          train_bm_label_np,train_am_label_np))
        '''
        train_set.append((train_raw_np,train_bm_label_np,train_am_label_np))
        

    for j in range(2,6):
        idx_test=str(j).rjust(2,'0')

        test_raw_name='S'+idx_test+"_"+str(0)+'raw'+'_'+str(size)+'.nii.gz'
        test_bm_label_name='S'+idx_test+"_"+str(0)+'bm_label'+'_'+str(size)+'.nii.gz'
        test_am_label_name='S'+idx_test+"_"+str(0)+'am_label'+'_'+str(size)+'.nii.gz'

        test_raw=nibabel.load(img_path+test_raw_name)
        test_bm_label=nibabel.load(img_path+test_bm_label_name)
        test_am_label=nibabel.load(img_path+test_am_label_name)

        test_raw_np=norm_to_0_1(test_raw.get_fdata())
        test_bm_label_np=norm_to_0_1(test_bm_label.get_fdata())
        test_am_label_np=test_am_label.get_fdata()

        '''
        test_set.append((fixed_np,test_raw_np,
                          fixed_np_amlabel,
                          test_bm_label_np,test_am_label_np))
        '''
        test_set.append((test_raw_np,test_bm_label_np,test_am_label_np))
            
    for a in range(6,11):
        idx_val=str(a).rjust(2,'0')

        val_raw_name='S'+idx_val+"_"+str(0)+'raw'+'_'+str(size)+'.nii.gz'
        val_bm_label_name='S'+idx_val+"_"+str(0)+'bm_label'+'_'+str(size)+'.nii.gz'
        val_am_label_name='S'+idx_val+"_"+str(0)+'am_label'+'_'+str(size)+'.nii.gz'

        val_raw=nibabel.load(img_path+val_raw_name)
        val_bm_label=nibabel.load(img_path+val_bm_label_name)
        val_am_label=nibabel.load(img_path+val_am_label_name)

        val_raw_np=norm_to_0_1(val_raw.get_fdata())
        val_bm_label_np=norm_to_0_1(val_bm_label.get_fdata())
        val_am_label_np=val_am_label.get_fdata()

        '''
        val_set.append((fixed_np,val_raw_np,
                          fixed_np_amlabel,
                          val_bm_label_np,val_am_label_np))
        '''
        val_set.append((val_raw_np,val_bm_label_np,val_am_label_np))
        
    np.save('./dataset/LPBA40_fixed_'+str(size)+'.npy', fixed_np)
    np.save('./dataset/LPBA40_fixed_amlabel_'+str(size)+'.npy', fixed_np_amlabel)
        
    np.save('./dataset/LPBA40_train_'+str(size)+'.npy', train_set)
    np.save('./dataset/LPBA40_test_'+str(size)+'.npy', test_set)
    np.save('./dataset/LPBA40_val_'+str(size)+'.npy', val_set)
    
    return


def preprocess_CC359_96():
    for i in range(1,359+1):
        idx=str(i).rjust(4,'0')
        raw_name='CC'+str(idx)+".nii.gz"
        bm_label_name='CC'+str(idx)+"_staple.nii.gz"
        am_label_name='CC'+str(idx)+"_wmstaple.nii.gz"

        raw_img=nibabel.load('./dataset/CC359/Original/'+raw_name)
        bm_label_img=nibabel.load('./dataset/CC359/Skull-stripping-masks/Silver-standard-STAPLE/STAPLE/'+bm_label_name)
        am_label_img=nibabel.load('./dataset/CC359/White-matter-masks/STAPLE-wm/STAPLE-wm/'+am_label_name)

        voxel_size = [2.666, 2.666, 2.666]
        resampled_raw_img = nibabel.processing.resample_to_output(raw_img, voxel_size,order=1)
        resampled_bm_label_img = nibabel.processing.resample_to_output(bm_label_img, voxel_size,order=0)
        resampled_am_label_img = nibabel.processing.resample_to_output(am_label_img, voxel_size,order=0)
        #resampled_skulled_img = nibabel.processing.resample_to_output(skulled_img, voxel_size,order=3)

        resampled_raw_np=image_to_square_v2(resampled_raw_img.get_fdata(),96)
        resampled_bm_label_np=image_to_square_v2(resampled_bm_label_img.get_fdata(),96)
        resampled_am_label_np=image_to_square_v2(resampled_am_label_img.get_fdata(),96)

        resampled_bm_label_np[resampled_bm_label_np>=0.5]=1.0
        resampled_bm_label_np[resampled_bm_label_np<0.5]=0.0
        
        resampled_am_label_np[resampled_am_label_np>=0.5]=1.0
        resampled_am_label_np[resampled_am_label_np<0.5]=0.0

        resampled_skulled_np=resampled_raw_np*resampled_bm_label_np

        clipped_img = nib.Nifti1Image(resampled_raw_np, resampled_raw_img.affine, nib.Nifti1Header())
        clipped_bm = nib.Nifti1Image(resampled_bm_label_np, resampled_bm_label_img.affine, nib.Nifti1Header())
        clipped_am = nib.Nifti1Image(resampled_am_label_np, resampled_am_label_img.affine, nib.Nifti1Header())
        clipped_skulled = nib.Nifti1Image(resampled_skulled_np, resampled_raw_img.affine, nib.Nifti1Header())

        img_path='./dataset/CC359_96/'

        raw_name_save = 'S'+str(idx)+"_"+str(0)+"raw"+"_"+str(96)+".nii.gz"
        bm_label_name_save = 'S'+str(idx)+"_"+str(0)+"bm_label"+"_"+str(96)+".nii.gz"
        am_label_name_save = 'S'+str(idx)+"_"+str(0)+"am_label"+"_"+str(96)+".nii.gz"
        skulled_name_save = 'S'+str(idx)+"_"+str(0)+"skulled"+"_"+str(96)+".nii.gz"

        nib.save(clipped_img, img_path+raw_name_save)
        nib.save(clipped_bm, img_path+bm_label_name_save)
        nib.save(clipped_am, img_path+am_label_name_save)
        nib.save(clipped_skulled, img_path+skulled_name_save)

    return


def get_dataset_CC359_96(size):
    train_set=[]
    test_set=[]
    val_set=[]

    img_path='./dataset/CC359_96/'
    fixed_name='S0001_0skulled'+"_"+str(size)+'.nii.gz'
    fixed_img=nibabel.load(img_path+fixed_name)
    fixed_np=norm_to_0_1(fixed_img.get_fdata())
    
    fixed_name_amlabel='S0001_0am_label'+"_"+str(size)+'.nii.gz'
    fixed_amlabel=nibabel.load(img_path+fixed_name_amlabel)
    fixed_np_amlabel=fixed_amlabel.get_fdata()
    
    
    test_name=loop_func(2,11)+loop_func(120,129)+loop_func(240,249)
    val_name=loop_func(12,21)+loop_func(130,139)+loop_func(250,259)
    train_name=loop_func(22,119)+loop_func(140,239)+loop_func(260,359)
    
    for idx_val in val_name:
        val_raw_name='S'+idx_val+"_"+str(0)+'raw'+'_'+str(size)+'.nii.gz'
        val_bm_label_name='S'+idx_val+"_"+str(0)+'bm_label'+'_'+str(size)+'.nii.gz'
        val_am_label_name='S'+idx_val+"_"+str(0)+'am_label'+'_'+str(size)+'.nii.gz'

        val_raw=nibabel.load(img_path+val_raw_name)
        val_bm_label=nibabel.load(img_path+val_bm_label_name)
        val_am_label=nibabel.load(img_path+val_am_label_name)

        val_raw_np=norm_to_0_1(val_raw.get_fdata())
        val_bm_label_np=norm_to_0_1(val_bm_label.get_fdata())
        val_am_label_np=val_am_label.get_fdata()
        
        
        '''
        val_set.append((fixed_np,val_raw_np,
                          fixed_np_amlabel,
                          val_bm_label_np,val_am_label_np))
        '''
        val_set.append((val_raw_np,val_bm_label_np,val_am_label_np))
             
            
    for idx_test in test_name:

        test_raw_name='S'+idx_test+"_"+str(0)+'raw'+'_'+str(size)+'.nii.gz'
        test_bm_label_name='S'+idx_test+"_"+str(0)+'bm_label'+'_'+str(size)+'.nii.gz'
        test_am_label_name='S'+idx_test+"_"+str(0)+'am_label'+'_'+str(size)+'.nii.gz'

        test_raw=nibabel.load(img_path+test_raw_name)
        test_bm_label=nibabel.load(img_path+test_bm_label_name)
        test_am_label=nibabel.load(img_path+test_am_label_name)

        test_raw_np=norm_to_0_1(test_raw.get_fdata())
        test_bm_label_np=norm_to_0_1(test_bm_label.get_fdata())
        test_am_label_np=test_am_label.get_fdata()
        
        '''
        test_set.append((fixed_np,test_raw_np,
                          fixed_np_amlabel,
                          test_bm_label_np,test_am_label_np))
        '''
        test_set.append((test_raw_np,test_bm_label_np,test_am_label_np))
        
     
    for idx_train in train_name:
            
        train_raw_name='S'+idx_train+"_"+str(0)+'raw'+'_'+str(size)+'.nii.gz'
        train_bm_label_name='S'+idx_train+"_"+str(0)+'bm_label'+'_'+str(size)+'.nii.gz'
        train_am_label_name='S'+idx_train+"_"+str(0)+'am_label'+'_'+str(size)+'.nii.gz'

        train_raw=nibabel.load(img_path+train_raw_name)
        train_bm_label=nibabel.load(img_path+train_bm_label_name)
        train_am_label=nibabel.load(img_path+train_am_label_name)

        train_raw_np=norm_to_0_1(train_raw.get_fdata())
        train_bm_label_np=norm_to_0_1(train_bm_label.get_fdata())
        train_am_label_np=train_am_label.get_fdata()
        
        
        '''
        train_set.append((fixed_np,train_raw_np,
                          fixed_np_amlabel,
                          train_bm_label_np,train_am_label_np))
        '''
        train_set.append((train_raw_np,train_bm_label_np,train_am_label_np))
            
    
    np.save('./dataset/CC359_fixed_'+str(size)+'.npy', fixed_np)
    np.save('./dataset/CC359_fixed_amlabel_'+str(size)+'.npy', fixed_np_amlabel)
    
    np.save('./dataset/CC359_train_'+str(size)+'.npy', train_set)
    np.save('./dataset/CC359_test_'+str(size)+'.npy', test_set)
    np.save('./dataset/CC359_val_'+str(size)+'.npy', val_set)
    
    return


def IBSR_img2nii():
    for i in range(1,18+1):
        idx=str(i).rjust(2,'0')

        raw_name='IBSR'+"_"+str(idx)+"_ana.img"
        am_label_name='IBSR'+"_"+str(idx)+"_segTRI_ana.img"


        raw_name_nii='IBSR'+"_"+str(idx)+"_ana.nii.gz"
        am_label_name_nii='IBSR'+"_"+str(idx)+"_segTRI_ana.nii.gz"

        raw_path='./dataset/IBSR/'+'IBSR'+"_"+str(idx)+"/"+"images/analyze/"
        am_path='./dataset/IBSR/'+'IBSR'+"_"+str(idx)+"/"+"segmentation/analyze/"
        
        raw_img=sitk.ReadImage(raw_path+raw_name)
        am_label_img=sitk.ReadImage(am_path+am_label_name)
        

        sitk.WriteImage(raw_img,raw_path+raw_name_nii)
        sitk.WriteImage(am_label_img,am_path+am_label_name_nii)

    return

def reorient_affine(affine):
    
    affine_ori=affine.copy()
    
    affine_ori[0][0]=affine_ori[0][0]*-1
    affine_ori[2][1]=affine_ori[1][1]*-1
    affine_ori[1][1]=0
    affine_ori[1][2]=affine_ori[2][2]
    affine_ori[2][2]=0
    
    return affine_ori

def IBSR_reorient():
    for i in range(1,18+1):
        idx=str(i).rjust(2,'0')


        raw_name_nii='IBSR'+"_"+str(idx)+"_ana.nii.gz"
        am_label_name_nii='IBSR'+"_"+str(idx)+"_segTRI_ana.nii.gz"
        
        raw_name_nii_reorient='IBSR'+"_"+str(idx)+"_ana_reorient.nii.gz"
        am_label_name_nii_reorient='IBSR'+"_"+str(idx)+"_segTRI_ana_reorient.nii.gz"

        raw_path='./dataset/IBSR/'+'IBSR'+"_"+str(idx)+"/"+"images/analyze/"
        am_path='./dataset/IBSR/'+'IBSR'+"_"+str(idx)+"/"+"segmentation/analyze/"
        
        raw_img=nibabel.load(raw_path+raw_name_nii)
        am_label_img=nibabel.load(am_path+am_label_name_nii)
        
        new_affine=reorient_affine(raw_img.affine)
        
        raw_img_result=nib.Nifti1Image(raw_img.get_fdata(), new_affine, nib.Nifti1Header())
        am_label_img_result=nib.Nifti1Image(am_label_img.get_fdata(), new_affine, nib.Nifti1Header())
        

        nib.save(raw_img_result,raw_path+raw_name_nii_reorient)
        nib.save(am_label_img_result,am_path+am_label_name_nii_reorient)

    return

def IBSR_get_bm_am():
    for i in range(1,18+1):
        idx=str(i).rjust(2,'0')
        
        raw_name_nii_reorient='IBSR'+"_"+str(idx)+"_ana_reorient.nii.gz"
        TRI_label_name_nii_reorient='IBSR'+"_"+str(idx)+"_segTRI_ana_reorient.nii.gz"

        raw_path='./dataset/IBSR/'+'IBSR'+"_"+str(idx)+"/"+"images/analyze/"
        am_path='./dataset/IBSR/'+'IBSR'+"_"+str(idx)+"/"+"segmentation/analyze/"
        
        raw_img=nibabel.load(raw_path+raw_name_nii_reorient)
        TRI_label_img=nibabel.load(am_path+TRI_label_name_nii_reorient)
        
        raw_img_np=raw_img.get_fdata()
        TRI_label_np=TRI_label_img.get_fdata()
        
        bm_label_np=TRI_label_np.copy()
        bm_label_np[bm_label_np!=0]=1.0

        am_label_np=TRI_label_np.copy()
        am_label_np[am_label_np!=3.0]=0.0
        am_label_np[am_label_np==3.0]=1.0
        
        skulled_np=raw_img_np*bm_label_np
        
        raw_name='IBSR'+"_"+str(idx)+"_ana_raw.nii.gz"
        skulled_name='IBSR'+"_"+str(idx)+"_ana_skulled.nii.gz"
        bm_name='IBSR'+"_"+str(idx)+"_ana_bm.nii.gz"
        am_name='IBSR'+"_"+str(idx)+"_ana_am.nii.gz"
        
        raw_result=nib.Nifti1Image(raw_img_np, raw_img.affine, nib.Nifti1Header())
        skulled_result=nib.Nifti1Image(skulled_np, raw_img.affine, nib.Nifti1Header())
        bm_result=nib.Nifti1Image(bm_label_np, TRI_label_img.affine, nib.Nifti1Header())
        am_result=nib.Nifti1Image(am_label_np, TRI_label_img.affine, nib.Nifti1Header())
        
        
        nib.save(raw_result,raw_path+raw_name)
        nib.save(skulled_result,raw_path+skulled_name)
        nib.save(bm_result,am_path+bm_name)
        nib.save(am_result,am_path+am_name)
        

    return

def preprocess_IBSR_96():
    for i in range(1,18+1):
        idx=str(i).rjust(2,'0')

        raw_name_nii='IBSR'+"_"+str(idx)+"_ana_raw.nii.gz" 
        bm_label_name_nii='IBSR'+"_"+str(idx)+"_ana_bm.nii.gz"
        am_label_name_nii='IBSR'+"_"+str(idx)+"_ana_am.nii.gz"
        skulled_name_nii='IBSR'+"_"+str(idx)+"_ana_skulled.nii.gz"
        
        raw_path='./dataset/IBSR/'+'IBSR'+"_"+str(idx)+"/"+"images/analyze/"
        am_path='./dataset/IBSR/'+'IBSR'+"_"+str(idx)+"/"+"segmentation/analyze/"

        raw_img=nibabel.load(raw_path+raw_name_nii)
        bm_label_img=nibabel.load(am_path+bm_label_name_nii)
        am_label_img=nibabel.load(am_path+am_label_name_nii)
        skulled_img=nibabel.load(raw_path+skulled_name_nii)
        
        voxel_size = [2.666, 2.666, 2.666]
        resampled_raw_img = nibabel.processing.resample_to_output(raw_img, voxel_size,order=1)
        resampled_bm_label_img = nibabel.processing.resample_to_output(bm_label_img, voxel_size,order=0)
        resampled_am_label_img = nibabel.processing.resample_to_output(am_label_img, voxel_size,order=0)
        resampled_skulled_img = nibabel.processing.resample_to_output(skulled_img, voxel_size,order=1)
        
        resampled_raw_np=image_to_square_v2(resampled_raw_img.get_fdata(),96)
        resampled_bm_label_np=image_to_square_v2(resampled_bm_label_img.get_fdata(),96)
        resampled_am_label_np=image_to_square_v2(resampled_am_label_img.get_fdata(),96)
        resampled_skulled_np=image_to_square_v2(resampled_skulled_img.get_fdata(),96)
        
        
        clipped_img = nib.Nifti1Image(resampled_raw_np, resampled_raw_img.affine, nib.Nifti1Header())
        clipped_bm = nib.Nifti1Image(resampled_bm_label_np, resampled_bm_label_img.affine, nib.Nifti1Header())
        clipped_am = nib.Nifti1Image(resampled_am_label_np, resampled_am_label_img.affine, nib.Nifti1Header())
        clipped_skulled = nib.Nifti1Image(resampled_skulled_np, resampled_skulled_img.affine, nib.Nifti1Header())
    
        img_path='./dataset/IBSR_96/'

        raw_name_save = 'S'+str(idx)+"_"+str(0)+"raw"+"_"+str(96)+".nii.gz"
        bm_label_name_save = 'S'+str(idx)+"_"+str(0)+"bm_label"+"_"+str(96)+".nii.gz"
        am_label_name_save = 'S'+str(idx)+"_"+str(0)+"am_label"+"_"+str(96)+".nii.gz"
        skulled_name_save = 'S'+str(idx)+"_"+str(0)+"skulled"+"_"+str(96)+".nii.gz"

        nib.save(clipped_img, img_path+raw_name_save)
        nib.save(clipped_bm, img_path+bm_label_name_save)
        nib.save(clipped_am, img_path+am_label_name_save)
        nib.save(clipped_skulled, img_path+skulled_name_save)
    
    return

def get_TEST_dataset_IBSR_96(size):

    test_set=[]

    img_path='./dataset/IBSR_96/'

    for i in range(1,18+1):
        idx_test=str(i).rjust(2,'0')

        test_raw_name='S'+idx_test+"_"+str(0)+'raw'+'_'+str(size)+'.nii.gz'
        test_bm_label_name='S'+idx_test+"_"+str(0)+'bm_label'+'_'+str(size)+'.nii.gz'
        test_am_label_name='S'+idx_test+"_"+str(0)+'am_label'+'_'+str(size)+'.nii.gz'

        test_raw=nibabel.load(img_path+test_raw_name)
        test_bm_label=nibabel.load(img_path+test_bm_label_name)
        test_am_label=nibabel.load(img_path+test_am_label_name)

        test_raw_np=norm_to_0_1(test_raw.get_fdata())
        test_bm_label_np=norm_to_0_1(test_bm_label.get_fdata())
        test_am_label_np=test_am_label.get_fdata()

        '''
        test_set.append((fixed_np,test_raw_np,
                          fixed_np_amlabel,
                          test_bm_label_np,test_am_label_np))
        '''
        test_set.append((test_raw_np,test_bm_label_np,test_am_label_np))
            
    np.save('./dataset/IBSR_test_'+str(size)+'.npy', test_set)
    
    return



def set_LPBA40_96():
    LPBA40_img2nii()
    preprocess_LPBA40_96()
    get_dataset_LPBA40_96(96)
    return

def set_CC359_96():
    preprocess_CC359_96()
    get_dataset_CC359_96(96)
    return


def set_IBSR_96():
    IBSR_img2nii()
    IBSR_reorient()
    IBSR_get_bm_am()
    preprocess_IBSR_96()
    get_TEST_dataset_IBSR_96(96)
    return

if __name__ == "__main__":
    set_LPBA40_96()
    set_CC359_96()
    set_IBSR_96()





