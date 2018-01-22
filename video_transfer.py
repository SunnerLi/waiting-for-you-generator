import _init_paths
from photo_transfer import Unnormalize
from torch.autograd import Variable
from torchvision import transforms
from model import CustomCycleGAN
import torchvision_extend.transforms as my_transforms
import torchvision.transforms.functional as F
import torchvision_extend.data as mData
import numpy as np
import subprocess
import argparse
import torch
import cv2
import os

"""
    This code can do the transfer toward the video
"""

VIDEO_INPUT_FOLDER = './input_temp_video'
VIDEO_OUTPUT_FOLDER = './output_temp_video'
counter = 0

def transferImage(model, img, output_folder = './', result_img_name = 'result.png'):
    global counter

    # Make as variable and normalized
    img = img.float()
    mean_list = [127.5, 127.5, 127.5]
    std_list = [127.5, 127.5, 127.5]
    for t in img:
        t = F.normalize(t, mean_list, std_list)
    img = Variable(img.float()).cuda()

    # Transfer    
    img = model(img)
    result_tensor = img

    # Save
    for img in result_tensor:
        img = img.data.cpu().numpy()
        img = torch.from_numpy(img).float()
        img = Unnormalize(img, mean_list, std_list)
        img = img.transpose(0, 1).transpose(1, 2)
        img = img.numpy()
        img = (img).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, 'frame_%d.png' % counter), img)
        counter += 1

if __name__ == '__main__':
    # Deal with parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='waiting_for_you.mp4', dest='input', help='The name of input image')
    parser.add_argument('--model', type=str, default='./model/', dest='model', help='The path of model')
    parser.add_argument('--output', type=str, default='result.mp4', dest='output', help='The name of output image')
    args = parser.parse_args()
    input_video_name = args.input
    model_path = args.model
    output_video_name = args.output

    # Check the folder is exist
    if not os.path.exists(model_path):
        print('You should train model first...')
        exit()
    if not os.path.exists(VIDEO_INPUT_FOLDER):
        os.mkdir(VIDEO_INPUT_FOLDER)
    if not os.path.exists(VIDEO_OUTPUT_FOLDER):
        os.mkdir(VIDEO_OUTPUT_FOLDER)

    # Decode video into images
    in_args = ['ffmpeg', '-i', input_video_name, '%s/frame_%%d.jpg' % VIDEO_INPUT_FOLDER]
    subprocess.call(" ".join(in_args), shell=True)

    # Deal with image name
    img_name_list = os.listdir(VIDEO_INPUT_FOLDER)
    for name in img_name_list:
        idx = int(name.split('_')[-1].split('.')[0])
        new_name = ('%9d' % idx).replace(' ', '0')
        new_name = name.split('_')[0] + '_' + new_name + '.jpg'
        os.rename(os.path.join(VIDEO_INPUT_FOLDER, name), os.path.join(VIDEO_INPUT_FOLDER, new_name))
        
    # work
    subprocess.call(["cp", os.path.join(VIDEO_INPUT_FOLDER, new_name), os.path.join(VIDEO_OUTPUT_FOLDER, new_name[:-4] + '.png')])
    dataset = mData.WaitTensorDataset(
        VIDEO_INPUT_FOLDER, VIDEO_OUTPUT_FOLDER,
        transform = transforms.Compose([
                my_transforms.Rescale((160, 320)),
                my_transforms.ToTensor(),
            ])
    )
    loader = mData.WaitDataLoader(dataset = dataset, batch_size=16, shuffle=False, num_workers = 2)
    model = CustomCycleGAN(model_folder = './cycleGAN_model/')
    model.cuda()
    model.load()
    
    for i, (batch_img, _) in enumerate(loader):
        print('Process progress - %.2f' % (i/loader.iter_num * 100))
        transferImage(model, batch_img, output_folder = VIDEO_OUTPUT_FOLDER, result_img_name = os.path.join(VIDEO_OUTPUT_FOLDER, 'frame_%d.png' % i))

    # Encode as output video
    frame_per_second = 30
    out_args = ['ffmpeg', '-i', '%s/frame_%%d.png' % VIDEO_OUTPUT_FOLDER, '-f', 'mp4', '-q:v', '0', '-vcodec', 'mpeg4', '-r', str(frame_per_second), output_video_name]
    subprocess.call(" ".join(out_args), shell=True)