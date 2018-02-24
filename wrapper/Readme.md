# Torchvision_sunner
### The flexible extension of torchvision

[![Packagist](https://img.shields.io/badge/Pytorch-0.3.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Torchvision-0.2.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()
[![Packagist](https://img.shields.io/badge/OpenCV-3.1.0-brightgreen.svg)]()
[![Packagist](https://img.shields.io/badge/skImage-0.13.1-green.svg)]()

Motivation
---
In pytorch, the common dataset can be load in an easy way. It also provides the `TensorDataset` to form the dataset. However, if we want to custom our unique image folder, or we want to load the muultiple image, the original methods cannot complete this work. In this package, you can load multiple images in an easy way!    

Install
---
1. download `torchvision_sunner` folder
2. put it in your current folder
3. import library and done!

Dataset
---
The examples will use waiting-for-you dataset as example. You can find the dataset [here](https://www.dropbox.com/s/cbuwbrehgglebhp/waiting_for_you_dataset.zip?dl=0). Download and extract it before you run the example. You can use your own data too! Just give the path of the folder, the module will load the image automatically!         


Usage
---
Import library first
```python
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
```

Form dataset:
```python
# Load single image folder
dataset = sunnerData.ImageDataset(
    root_list = ['./waiting_for_you_dataset/real_world'],
)

# Load multiple image folder
dataset = sunnerData.ImageDataset(
    root_list = ['./waiting_for_you_dataset/real_world', './waiting_for_you_dataset/wait'],
)
```

Extra transpose method
```python
dataset = sunnerData.ImageDataset(
    root_list = ['./waiting_for_you_dataset/real_world', './waiting_for_you_dataset/wait'],
    transform = transforms.Compose([
        sunnertransforms.Rescale((160, 320)),
        sunnertransforms.ToTensor(),

        # BHWC -> BCHW
        sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        sunnertransforms.Normalize([127., 127., 127.], [127., 127., 127.])
    ]) 
)
```

The Z normalization is also supported in default! Just don't give any parameter.
```python
# Assume the range of original distribution is [0, 255]
sunnertransforms.Normalize(     # The range will squeeze in [-1.007, 1.007]
          [127., 127., 127.], [127., 127., 127.]
)   
sunnertransforms.Normalize()    # The range will squeeze in [-1, 1]
```

The wrapper of usual `Dataloader`, but it can know the whole number of batch!
```python
loader = sunnerData.ImageLoader(dataset, batch_size=32, shuffle=False, num_workers = 2)
```

Transfer the tensor into numpy, and it's similar to the usage of `ImageDataset`!
```python
batch_img = sunnertransforms.tensor2Numpy(batch_img, transform = transforms.Compose([
    sunnertransforms.UnNormalize([127., 127., 127.], [127., 127., 127.]),
    sunnertransforms.UnNormalize(),     # usual unnormalization
    sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
]))
```
You can check example3 to see the detail usage of normalization without parameters.    

Notice
---
* This package provides two backend image processing library working: opencv and skimage. Since the opencv can show the continuous image easily, the default library we use is opencv. On the contrary, the installation of opencv is tedious. You can choose skimage to become the backend library while it can be easily installed. 
```python
dataset = sunData.ImageDataset(
    root_list = ['./waiting_for_you_dataset/real_world'],
    use_cv = False,                                          # Revised here
    transform = transforms.Compose([
        suntransforms.Rescale((160, 320), use_cv = False),   # Revised here too
        suntransforms.ToTensor(),
    ]) 
)
```
However, you should change the channel order before you print the image:
```python
batch_img = cv2.cvtColor(batch_img, cv2.COLOR_RGB2BGR)
cv2.imshow('show_window', batch_img)
```

         
* `tensor2Numpy` is function, and it just deals with single batch image. The detailed usage can be referred in example script.    
*  This project doesn't provides PyPI installation approach.    