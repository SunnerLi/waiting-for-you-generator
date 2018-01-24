## Waiting for you Generator (Style Transfer with Mask-CycleGAN)
[![Packagist](https://img.shields.io/badge/Pytorch-0.3.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Torchvision-0.2.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()

![](https://raw.githubusercontent.com/SunnerLi/waiting-for-you-generator/master/img/usual_cycleGAN_result/9_3500.png?token=AK99R3VAW0xMvrh7HeGB41M7zIomwRtEks5acW9dwA%3D%3D)

Motivation
---
In 18th January 2018, the famous singer Jay-Chou released the new song - waiting for you. In the MV of this song, the love story is described by the smooth screen. Moreover, the actual site he filming this video is National Taiwan Normal University which differ from my school.     

Contribution
---
1. use cycleGAN and mask-cycleGAN to transfer the image and video into waiting-for-you image space
2. provide `ImageDataset` and `ImageLoader` which is more flexible to the multiple image folder.    

Abstraction
---
In the video of the song, there're some text covering the screen. In this project, the mask cycle generative adversarial network (Mask-CycleGAN) structure is purposed to solve this problem. Not only using usual cycleGAN to do the style transfer, but also adopting a mask to shield the influence of these texts. By using our model, you can create your own waiting for you video, and invest the unique touching for your own!!!      

Cycle Structure
---
ssssss    

Environment
---
OS: Ubuntu 16.04     
Dataset: MSCOCO 2014 (real space) and the shot of waiting for you MV (waiting space)    

Usage
---
You should download the dataset by yourself:
```
# Prepare training data for real space
$ wget https://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip

# Prepare training data for waiting space
$ python split.py

# Training
$ python train.py

# Transfer the photo
$ python photo_transfer.py --input ./img/nctu_sport_field.png --model ./model_mask_cycleGAN/ --output ./img/result.png

# Transfer the video
$ python video_transfer.py --input your_love_video.mp4 --model model_mask_cycleGAN --output result.mp4
```

Result
---
![](https://raw.githubusercontent.com/SunnerLi/waiting-for-you-generator/master/img/article/loss_merge.png?token=AK99RwEToXKHPcOQTPCGVTmVHP8_Bg91ks5acXBRwA%3D%3D)
The above image shows the loss curve. As you can see, both idea can converge at the end. You can refer to the report article to get the detail of my result.    