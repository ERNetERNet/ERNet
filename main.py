from train import *

if __name__ == "__main__":
    
    train_set_name="LPBA40_train_sample.npy"    # "LPBA40_train_96.npy"   "CC359_train_96.npy"
    val_set_name="LPBA40_val_sample.npy"        # "LPBA40_val_96.npy"   "CC359_val_96.npy"
    test_set_name="LPBA40_test_sample.npy"      # "LPBA40_test_96.npy"  "CC359_test_96.npy"   "IBSR_test_96.npy"
    dice_label="LPBA40"                     # "LPBA40"   "CC359" 
    fixed_set_name="LPBA40"                 # "LPBA40"   "CC359"
    reg_loss_name="NCC"                     # "NCC"  "GCC"  "MSE"
    gamma=10
    lamda_mask=1.0
    mask_smooth_loss_func=first_Grad("l2")
    ext_stage=5
    reg_stage=5
    if_train_aug=True
    batch_size=1
    img_size=96
    num_epochs=1000
    learning_rate=0.000001
    save_every_epoch=1
    save_start_epoch=0
    model_name=ERNet(img_size, ext_stage , reg_stage, gamma)
    
    train_96(img_size,
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
             save_start_epoch)