import _init_paths
from torch.autograd import Variable
from torchvision import transforms
from cycle_gan import CycleGAN
from skimage import io
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms.functional as F
import numpy as np
import subprocess
import argparse
import torch
import os

"""
    This code can do the transfer toward the video
"""

VIDEO_INPUT_FOLDER = './input_temp_video'
VIDEO_OUTPUT_FOLDER = './output_temp_video'
counter = 0

def transferImage(model, img, output_folder = './', result_img_name = 'result.png'):
    global counter

    data = {'B': img}
    model.set_input(data)
    model.test()
    result_tensor = model.fake_B
    result_tensor = sunnertransforms.tensor2Numpy(result_tensor, transform = transforms.Compose([
        sunnertransforms.UnNormalize(),
        sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
    ]))

    # Save
    for img in result_tensor:
        img = (img).astype(np.uint8)
        io.imsave(os.path.join(output_folder, 'frame_%d.png' % counter), img)
        counter += 1

if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------------------------
    # Deal with parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='waiting_for_you.mp4', dest='input', help='The name of input image')
    parser.add_argument('--model', type=str, default='./model/', dest='model', help='The path of model')
    parser.add_argument('--output', type=str, default='result.mp4', dest='output', help='The name of output image')
    args = parser.parse_args()
    input_video_name = args.input
    model_path = args.model
    output_video_name = args.output

    # ---------------------------------------------------------------------------------------------------------
    # Check the folder is exist
    if not os.path.exists(model_path):
        print('You should train model first...')
        exit()
    if not os.path.exists(VIDEO_INPUT_FOLDER):
        os.mkdir(VIDEO_INPUT_FOLDER)
    if not os.path.exists(VIDEO_OUTPUT_FOLDER):
        os.mkdir(VIDEO_OUTPUT_FOLDER)

    # ---------------------------------------------------------------------------------------------------------
    # Decode video into images
    in_args = ['ffmpeg', '-i', input_video_name, '%s/frame_%%d.jpg' % VIDEO_INPUT_FOLDER]
    subprocess.call(" ".join(in_args), shell=True)

    # ---------------------------------------------------------------------------------------------------------
    # Deal with image name
    img_name_list = os.listdir(VIDEO_INPUT_FOLDER)
    for name in img_name_list:
        idx = int(name.split('_')[-1].split('.')[0])
        new_name = ('%9d' % idx).replace(' ', '0')
        new_name = name.split('_')[0] + '_' + new_name + '.jpg'
        os.rename(os.path.join(VIDEO_INPUT_FOLDER, name), os.path.join(VIDEO_INPUT_FOLDER, new_name))
        
    # ---------------------------------------------------------------------------------------------------------
    # work
    subprocess.call(["cp", os.path.join(VIDEO_INPUT_FOLDER, new_name), os.path.join(VIDEO_OUTPUT_FOLDER, new_name[:-4] + '.png')])
    dataset = sunnerData.ImageDataset(
        root_list = [VIDEO_INPUT_FOLDER], 
        use_cv = False,
        transform = transforms.Compose([
            sunnertransforms.Rescale((160, 320), use_cv = False),
            sunnertransforms.ToTensor(),
            sunnertransforms.ToFloat(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize()
        ])
    )
    loader = sunnerData.ImageLoader(dataset, batch_size=32, shuffle=False, num_workers = 2)
    model = CycleGAN(model_path, \
        isTrain = False, \
        input_channel = 3, \
        output_channel = 3, \
        base_filter = 32, \
        batch_size = 4, \
        use_dropout = False, \
        use_gpu = True)
    for i, batch_img in enumerate(loader):
        batch_img = torch.cat(batch_img, 0)
        print(batch_img.size())
        print('Process progress - %.2f' % (i/loader.iter_num * 100))
        transferImage(model, batch_img, output_folder = VIDEO_OUTPUT_FOLDER, \
            result_img_name = os.path.join(VIDEO_OUTPUT_FOLDER, 'frame_%d.png' % i))

    # ---------------------------------------------------------------------------------------------------------
    # Encode as output video
    frame_per_second = 30
    out_args = ['ffmpeg', '-i', '%s/frame_%%d.png' % VIDEO_OUTPUT_FOLDER, '-f', 'mp4', '-q:v', '0', '-vcodec', 'mpeg4', '-r', \
        str(frame_per_second), output_video_name]
    subprocess.call(" ".join(out_args), shell=True)

    # ---------------------------------------------------------------------------------------------------------
    # Remove temp folder
    import shutil
    shutil.rmtree(VIDEO_INPUT_FOLDER)
    shutil.rmtree(VIDEO_OUTPUT_FOLDER)