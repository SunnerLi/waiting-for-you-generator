## Waiting for you Generator (Style Transfer with Mask-CycleGAN)
[![Packagist](https://img.shields.io/badge/Pytorch-0.3.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Torchvision-0.2.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()

![](https://github.com/SunnerLi/waiting-for-you-generator/blob/original_structure/img/usual_cycleGAN_result/result.png)

Abstract
---
This is the another version of waiting for you generator. The branch use original structure to enhance the render performance. The structure of generator is ResNet-6, and the normalization technique is instance normalization. Most of the important, this branch adopt original GAN loss as the cost function.     

Dataset
---
In order to increase the shared semantic between different modality, this brance use our-collected dataset without MSCOCO dataset. There are two modalities in the dataset:    
* Real world modality: We capture the scan from two real-cover video. Additionally, the screen with much low brightness would be ignored in this image domain. [[link1](https://www.youtube.com/watch?v=HS-giu1EvWc)][[link2](https://www.youtube.com/watch?v=5aGRw0_gSxY)]    
* Waiting-for-you modality: We record the scan from the two MV that Jay chou released. In the second version, the screen with snow background would be ignored since it's in the real world modality. [[link1](https://www.youtube.com/watch?v=kfXdP7nZIiE)][[link2](https://www.youtube.com/watch?v=QQucPUfXUQQ)]     

Usage
---
The dataset is located in Dropbox platform (url: `https://www.dropbox.com/s/cbuwbrehgglebhp/waiting_for_you_dataset.zip?dl=0`). As the result, you should download the file before you start to train.

**Another point you should notice!** The parameter of transfer is changed. You can refer the following example:
```
# Training
$ python train.py

# Transfer the photo
$ python photo_transfer.py --input ./img/nctu_sport_field.png --model ./mask_cycleGAN_output/ --output result.png

# Transfer the video
$ python video_transfer.py --input waiting_for_you.mp4 --model ./mask_cycleGAN_output/ --output result.mp4
```

Render Result
---
The top picture has shown the result of usual CycleGAN, and the following shows the render result of Mask-CycleGAN:

![](https://github.com/SunnerLi/waiting-for-you-generator/blob/original_structure/img/mask_cycleGAN_result/bce_result.png)

We also adopt least square loss and train again! The following image shows the result of LS loss:

![](https://github.com/SunnerLi/waiting-for-you-generator/blob/original_structure/img/mask_cycleGAN_result/ls_.png)