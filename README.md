# StarGAN implementation

This code is an implementation of the paper 'StarGAN: Unified Generative Adversarial Networks
for Multi-Domain Image-to-Image Translation'. The algorithm is based on the following papers:

    Choi, Y., Choi, M., Kim, M., Ha, J. W., Kim, S., & Choo, J. (2018). Stargan: Unified generative adversarial networks for multi-domain image-to-image translation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 8789-8797).
    
# Structrue
![Picture1](https://user-images.githubusercontent.com/17133841/61397795-aba93800-a898-11e9-921f-557ba4680e67.png)

# Used dataset
Uploaded code is compatible with Large-scale CelebFaces Attributes (CelebA) dataset. Model used the __align&cropped images__ and __attribute annotation__ file.

    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    
# Instruction
Before proceeding, please set the 'Hyper_Parameter.py' according to your environment.

## Training

    python StarGAN.py
    
## Test

### Run 'ipython' in the model's directory.
### Run following command:
    import numpy as np
    from StarGAN import StarGAN
    
    new_StarGAN = StarGAN()
    new_StarGAN.Restore()

### Set the image file path list and attribute list like the following example:
    file_Name_List=[
        './Photo.jpg'
        ]
    attribute_Pattern_List=[
        np.array([1, 0, 0, 1, 0, 0]),
        np.array([0, 1, 0, 1, 0, 0]),
        np.array([0, 0, 1, 1, 0, 0]),
        np.array([1, 0, 0, 0, 0, 0]),
        np.array([1, 0, 0, 0, 1, 0]),
        np.array([0, 1, 0, 0, 1, 0])
        ]
        
### Run following command:
    new_StarGAN.Test(
        file_Name_List= file_Name_List,
        attribute_Pattern_List= attribute_Pattern_List,
        export_File_Name= 'Result'
        )
    
# Result
![Step_134000](https://user-images.githubusercontent.com/17133841/61399079-5f132c00-a89b-11e9-828f-677c7a135192.png)

![Result](https://user-images.githubusercontent.com/17133841/61399088-64707680-a89b-11e9-8ae6-7c0154d5f41e.png)
