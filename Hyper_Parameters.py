import numpy as np

attribute_File_Path = "D:/Simulation_Raw_Data/CelebA/list_attr_celeba.txt"
image_Files_Dir = "D:/Simulation_Raw_Data/CelebA/img"
select_Attribute_List = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young", "Pale_Skin"]   #If 'None', model use all attributes
image_Size = 128
initial_Filter_Count = 64   #For bottleneck
batch_Size = 16
gan_Loss_Type = 'WGAN'  #'WGAN' or 'LSGAN'
learning_Rate = 0.0001
extract_Dir = 'E:/GAN_Result/StarGAN'


test_File_Name_List = [
    'D:/Simulation_Raw_Data/CelebA/Img/188712.jpg',
    'D:/Simulation_Raw_Data/CelebA/Img/188673.jpg',
    'D:/Simulation_Raw_Data/CelebA/Img/189052.jpg',
    'D:/Simulation_Raw_Data/CelebA/Img/201436.jpg',
    'D:/Simulation_Raw_Data/CelebA/Img/201881.jpg',
    'D:/Simulation_Raw_Data/CelebA/Img/201434.jpg',
    ]

test_Attribute_Pattern_List = [
    np.array([1, 0, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0]),
    np.array([0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 0, 1, 0, 0]),
    np.array([0, 0, 0, 0, 1, 0]),
    np.array([0, 0, 0, 0, 0, 1]),            
    np.array([0, 1, 0, 1, 0, 0]),
    np.array([0, 1, 0, 0, 1, 0]),
    np.array([0, 0, 0, 1, 1, 0]),
    np.array([0, 0, 1, 1, 1, 0]),
    ]   #Each select_Attribute_List

test_Step = 1000
checkpoint_Step = 1000