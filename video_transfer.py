import _init_paths
from photo_transfer import Unnormalize
from torch.autograd import Variable
from model import CustomCycleGAN
import torchvision.transforms.functional as F
import numpy as np
import subprocess
import argparse
import torch
import cv2
import os

VIDEO_INPUT_FOLDER = './input_temp_video'
VIDEO_OUTPUT_FOLDER = './output_temp_video'

def transferImage(model, img, output_folder = './', result_img_name = 'result.png'):
    # Make as variable and normalized
    img = torch.from_numpy(img).float()
    img = img.transpose(1, 2).transpose(0, 1)
    mean_list = [127.5, 127.5, 127.5]
    std_list = [127.5, 127.5, 127.5]
    img = F.normalize(img, mean_list, std_list)
    img = img.numpy()[np.newaxis, :]
    img = Variable(torch.from_numpy(img).float()).cuda()

    # Transfer    
    img = model(img)

    # Save
    img = img.data.cpu().numpy()[0]
    img = torch.from_numpy(img).float()
    img = Unnormalize(img, mean_list, std_list)
    img = img.transpose(0, 1).transpose(1, 2)
    img = img.numpy()
    img = (img).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, result_img_name), img)

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
    in_args = ['ffmpeg', '-i', input_video_name, '%s/frame_%%d.png' % VIDEO_INPUT_FOLDER]
    subprocess.call(" ".join(in_args), shell=True)

    # work
    img_name_list = os.listdir(VIDEO_INPUT_FOLDER)
    model = CustomCycleGAN(model_folder = './cycleGAN_model/')
    model.cuda()
    model.load()
    for i, name in enumerate(img_name_list):
        print('Process progress - %.2f' % (i/len(img_name_list) * 100))
        img = cv2.imread(os.path.join(VIDEO_INPUT_FOLDER, name))
        transferImage(model, img, result_img_name = os.path.join(VIDEO_OUTPUT_FOLDER, name))
        break

    # Encode as output video
    frame_per_second = 30
    out_args = ['ffmpeg', '-i', '%s/frame_%%d.png' % VIDEO_OUTPUT_FOLDER, '-f', 'mp4', '-q:v', '0', '-vcodec', 'mpeg4', '-r', str(frame_per_second), output_video_name]
    subprocess.call(" ".join(out_args), shell=True)