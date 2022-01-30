from model import *
from utils import *

    
def train_96(img_size,
          reg_stage,
          ext_stage,
          train_set_name,
          val_set_name,
          test_set_name,
          batch_size,
          num_epochs,
          learning_rate,
          model_name,
          reg_loss_name,
          mask_smooth_loss_func,
          lamda_mask,
          gamma,
          save_every_epoch,
          dice_label,
          if_train_aug,
          fixed_set_name,
          save_start_epoch):
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = model_name.to(device)
    mask_smooth = mask_smooth_loss_func.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if reg_loss_name=="NCC":
        reg_loss_func=NCC().loss   
    elif reg_loss_name=="GCC":
        reg_loss_func=GCC()
    elif reg_loss_name=="MSE":
        reg_loss_func=nn.MSELoss().to(device)
        
    cur_path = os.getcwd()
    result_path=cur_path+'/result'
    
    loss_log_path=result_path+'/loss_log'
    create_folder(result_path,'loss_log')
    
    sample_img_path=result_path+'/sample_img'
    create_folder(result_path,'sample_img')
    
    model_save_path=result_path+'/model'
    create_folder(result_path,'model')
    
    model_str=str(model)[0:str(model).find("(")]
    lamda_mask_str=str(lamda_mask)
    gamma_str=str(gamma)
    lr_str=str(learning_rate)
    dataset_str=train_set_name[0:str(train_set_name).find(".")]
    
    ext_stage_str=str(ext_stage)
    reg_stage_str=str(reg_stage)
    
    if if_train_aug==True:
        modal_name=model_str+"_"+ext_stage_str+"_"+reg_stage_str+"_"+lamda_mask_str+"_"+gamma_str+"_"+lr_str+"_"+dataset_str+"_trainAug_true"
    else:
        modal_name=model_str+"_"+ext_stage_str+"_"+reg_stage_str+"_"+lamda_mask_str+"_"+gamma_str+"_"+lr_str+"_"+dataset_str
        
    
    modal_path=sample_img_path+"/"+modal_name
    create_folder(sample_img_path,modal_name)
    
    sample_o_path=modal_path+"/"+"o"
    sample_t_path=modal_path+"/"+"t"
    create_folder(modal_path,"o")
    create_folder(modal_path,"t")
    
    
    for i in range(int(ext_stage)):
        idx=i+1
        s_name="s_"+str(idx)
        s_mask_name="s_"+str(idx)+"_mask"
        sample_s_path=modal_path+"/"+s_name
        sample_s_mask_path=modal_path+"/"+s_mask_name
        
        create_folder(modal_path,s_name)
        create_folder(modal_path,s_mask_name)
    
    for q in range(int(reg_stage)):
        qdx=q+1
        r_name="r_"+str(qdx)
        r_grid_name="r_"+str(qdx)+"_grid"
        sample_r_path=modal_path+"/"+r_name
        sample_r_grid_path=modal_path+"/"+r_grid_name
        
        create_folder(modal_path,r_name)
        create_folder(modal_path,r_grid_name)
    if if_train_aug==True:
        modal_info="Model: {}    ext_stage: {}    reg_stage: {}    λ_mask: {}    γ: {}    lr: {}    dataset: {}    train_Agu: {}".format(model_str,
                                                                                       ext_stage_str,
                                                                                       reg_stage_str,
                                                                                       lamda_mask_str,
                                                                                       gamma_str,
                                                                                       lr_str,dataset_str,"true")
    else:
        modal_info="Model: {}    ext_stage: {}    reg_stage: {}    λ_mask: {}    γ: {}    lr: {}    dataset: {}".format(model_str,
                                                                                       ext_stage_str,
                                                                                       reg_stage_str,
                                                                                       lamda_mask_str,
                                                                                       gamma_str,
                                                                                       lr_str,dataset_str)
    
    create_log(modal_info,loss_log_path,modal_name)
    
    print (modal_info)

    train_loader=load_data_no_fix(train_set_name,batch_size)
    val_loader=load_data_no_fix(val_set_name,batch_size)
    test_loader=load_data_no_fix(test_set_name,batch_size)
    
    for epoch in range(num_epochs):
        
        total_loss_train=[]
        total_sim_loss_train=[]
        
        start=time.time()
        
        fixed_data=torch.from_numpy(np.load('./dataset/'+fixed_set_name+"_fixed_96.npy")).to(device).view(-1,1,img_size,img_size,img_size).float()
        fixed_label_am=torch.from_numpy(np.load('./dataset/'+fixed_set_name+"_fixed_amlabel_96.npy")).to(device).view(-1,1,img_size,img_size,img_size).float()
        fixed_data_val=torch.from_numpy(np.load('./dataset/'+fixed_set_name+"_fixed_96.npy")).to(device).view(-1,1,img_size,img_size,img_size).float()
        fixed_label_am_val=torch.from_numpy(np.load('./dataset/'+fixed_set_name+"_fixed_amlabel_96.npy")).to(device).view(-1,1,img_size,img_size,img_size).float()
        
        
        for i, x in enumerate(train_loader):
            
            moving_data,moving_label_bm,moving_label_am=x
            
            moving_data=moving_data.to(device).view(-1,1,img_size,img_size,img_size).float()
            moving_label=moving_label_bm.to(device).view(-1,1,img_size,img_size,img_size).float()
            
            
            if if_train_aug == True:
                
                
                degree=5
                voxel=5
                scale_min=0.98
                scale_max=1.02
                
                """
                degree=2
                voxel=2
                scale_min=0.98
                scale_max=1.02
                """
                
                moving_data,moving_label=train_aug(moving_data,moving_label,img_size,degree,voxel,scale_min,scale_max)
            
            optimizer.zero_grad()
            
            striped_list,mask_list,warped_list,grid_list= model(fixed_data,moving_data,if_train=True)
                
            sim_loss_train=reg_loss_func(warped_list[-1],fixed_data)
            
            mask_smooth_loss=lamda_mask*sum([mask_smooth(i) for i in mask_list])
            
            loss_train = sim_loss_train + mask_smooth_loss
            loss_train.backward()
            optimizer.step()
            
             
            total_loss_train.append(loss_train.item())
            total_sim_loss_train.append(sim_loss_train.item())
        
        ave_loss_train,std_loss_train=get_ave_std(total_loss_train)
        ave_sim_loss_train,std_sim_loss_train=get_ave_std(total_sim_loss_train)

        if epoch % save_every_epoch ==0:
            model.eval()
            with torch.no_grad():

                total_loss_val_sim_af=[]
                
                total_ext_val=[]
                total_dice_val=[]
                
                for j, y in enumerate(val_loader):
                    
                    moving_data_val,moving_label_bm_val,moving_label_am_val=y
                    
                    moving_data_val=moving_data_val.to(device).view(-1,1,img_size,img_size,img_size).float()
                    moving_label_bm_val=moving_label_bm_val.to(device).view(-1,1,img_size,img_size,img_size).float()
                    moving_label_am_val=moving_label_am_val.to(device).view(-1,1,img_size,img_size,img_size).float()
                    
                    striped_list_val,mask_list_val,warped_list_val,grid_list_val= model(fixed_data_val,moving_data_val,if_train=False)
                  
                    sim_loss_val=reg_loss_func(warped_list_val[-1],fixed_data_val)
                    
                    total_loss_val_sim_af.append(sim_loss_val.item())
                    
                    merged_mask_list_val=np.prod(mask_list_val)
                    ext_val=f1_dice_loss(merged_mask_list_val,moving_label_bm_val)
                    total_ext_val.append(ext_val)
                                
                    moving_label_am_skulled_val=merged_mask_list_val*moving_label_am_val
                    warped_label_am_skulled_val=F.grid_sample(moving_label_am_skulled_val, grid_list_val[-1],
                                                              mode="nearest",align_corners=True,padding_mode="zeros")

                    dice_val=compute_label_dice(fixed_label_am_val,warped_label_am_skulled_val,dice_label)

                    
                    total_dice_val.append(dice_val)
                    
                    
                ave_loss_val_sim,std_loss_val_sim=get_ave_std(total_loss_val_sim_af)

                ave_ext_val,std_ext_val=get_ave_std(total_ext_val)
                ave_dice_val,std_dice_val=get_ave_std(total_dice_val)

                loss_info="Epoch[{}/{}], All Training loss: {:.4f}/{:.4f} , Reg Training loss: {:.4f}/{:.4f} , Reg val loss: {:.4f}/{:.4f} , Dice_ext_val: {:.4f}/{:.4f}  ,  Dice_reg_val: {:.4f}/{:.4f}".format(epoch+1,num_epochs,
                ave_loss_train,std_loss_train,
                ave_sim_loss_train,std_sim_loss_train,
                ave_loss_val_sim,std_loss_val_sim,
                ave_ext_val,std_ext_val,
                ave_dice_val,std_dice_val
                )
                      
                print (loss_info)
                append_log(loss_info,loss_log_path,modal_name)
                
                
                if epoch>save_start_epoch:
                    
                    save_sample_any(epoch,"o",fixed_data_val,sample_o_path)
                    #save_nii_any(epoch,"o",fixed_data_val,sample_o_path)

                    save_sample_any(epoch,"t",moving_data_val,sample_t_path)
                    #save_nii_any(epoch,"t",moving_data_val,sample_t_path)

                    for t in range(int(ext_stage)):
                        tdx=t+1
                        s_name="s_"+str(tdx)
                        s_mask_name="s_"+str(tdx)+"_mask"

                        sample_s_path=modal_path+"/"+s_name
                        sample_s_mask_path=modal_path+"/"+s_mask_name

                        save_sample_any(epoch,s_name,striped_list_val[t],sample_s_path)
                        #save_nii_any(epoch,s_name,striped_list_val[t],sample_s_path)

                        save_sample_any(epoch,s_mask_name,mask_list_val[t],sample_s_mask_path)

                    for y in range(int(reg_stage)):
                        ydx=y+1
                        r_name="r_"+str(ydx)
                        sample_r_path=modal_path+"/"+r_name

                        save_sample_any(epoch,r_name,warped_list_val[y],sample_r_path)
                        #save_nii_any(epoch,r_name,warped_list_val[y],sample_r_path)


                    torch.save(model.state_dict(), os.path.join(model_save_path,modal_name+"_"+str(epoch)+".pth"))
        
    return    
    
    