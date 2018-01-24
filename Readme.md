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
For style transfer problem, there're pure-CNN based[1] method and GAN based[2] methods. However, in the video of this song, there're some text covering the screen. In this project, the mask cycle generative adversarial network (Mask-CycleGAN) structure is purposed to solve this problem which is revised from the usual CycleGAN[2]. Not only using usual CycleGAN to do the style transfer, but also adopting a mask to shield the influence of these texts. By using our model, you can create your own waiting for you video, and invest the unique touching for your own!!!      

Cycle Structure
---
![](https://raw.githubusercontent.com/SunnerLi/waiting-for-you-generator/master/img/article/cycle3.png?token=AK99R1l-LiJeIldqkd30pX0eQ_tsiW9tks5acXd8wA%3D%3D)    

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
$ python video_transfer.py --input waiting_for_you.mp4 --model ./model_mask_cycleGAN/ --output result.mp4
```

Result
---
![](https://raw.githubusercontent.com/SunnerLi/waiting-for-you-generator/master/img/article/loss_merge.png?token=AK99RwEToXKHPcOQTPCGVTmVHP8_Bg91ks5acXBRwA%3D%3D)
The above image shows the loss curve. As you can see, both idea can converge at the end. You can refer to the report article to get the detail of my result.    

Notice
---
Even pytorch can accept the arbitrary size of input image, the padding of convolution is rigid. It's recommend to resize as the time of (160 * 320), or some padding issue will raise.     

Reference
---
[1] Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, “Image Style Transfer Using Convolutional Neural Networks,” _In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, Las Vegas, Nevada, USA, 27–30 June, 2016, pp. 2414–2423.    

[2] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros, “Unpaired Image-to-Image Translation using CycleConsistent Adversarial Networks,” _In International Conference on Computer Vision (ICCV)_, Venice, Italy, 22- 29, October, 2017, pp. 2223–2232.