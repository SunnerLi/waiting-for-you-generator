import _init_paths
from mask_cycle_gan import MaskCycleGAN
from torchvision import transforms
from cycle_gan import CycleGAN
from util import saveImg
from config import *
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import pandas as pd
import numpy as np
import time
import os

# Define loss list
D_A_loss_list = []
D_B_loss_list = []
G_A_loss_list = []
G_B_loss_list = []
Cyc_A_loss_list = []
Cyc_B_loss_list = []

def updateLossList(errors):
    """
        Update the loss list with given error dict object

        Arg:    errors  - The dict object which given by model
    """
    global D_A_loss_list
    global D_B_loss_list
    global G_A_loss_list
    global G_B_loss_list
    global Cyc_A_loss_list
    global Cyc_B_loss_list

    # Append error
    D_A_loss_list.append(errors['D_A'])
    D_B_loss_list.append(errors['D_B'])
    G_A_loss_list.append(errors['G_A'])
    G_B_loss_list.append(errors['G_B'])
    Cyc_A_loss_list.append(errors['Cyc_A'])
    Cyc_B_loss_list.append(errors['Cyc_B'])

    # Store latest error csv
    df = pd.DataFrame({'D_A':[], 'D_B':[], 'G_A':[], 'G_B':[], 'Cyc_A':[], 'Cyc_B':[]})
    df['D_A'] = D_A_loss_list
    df['D_B'] = D_B_loss_list
    df['G_A'] = G_A_loss_list
    df['G_B'] = G_B_loss_list
    df['Cyc_A'] = Cyc_A_loss_list
    df['Cyc_B'] = Cyc_B_loss_list
    df.to_csv(csv_name)

def print_current_errors(step, errors, t):
    """
        Print the information of training

        Arg:    step    - The number of iterations
                errors  - The dict object which given by model
                t       - The time cost during batch_size * verbose_peroid
    """
    message = 'iteration: %d\t time: %.3f\t' % (step, t)
    for k, v in errors.items():
        message += '  %s: %.3f  ' % (k, v)
    print(message)
    with open(log_name, 'a') as f:
        f.write('%s\n' % message)

def train(model, loader, output_folder):
    """
        Main training function
    """
    global epoches
    global iteration
    
    total_step = 0
    while total_step < iteration:
        for data_a, data_b in loader:
            iter_start_time = time.time()
            model.set_input({'A': data_a, 'B': data_b}, use_gpu = True)
            model.optimize_parameters()

            if total_step % VERBOSE_PEROID == 0:
                errors = model.get_current_errors()
                model.save('latest')
                saveImg(model.get_current_visuals(), output_folder, str(total_step) + '.png')
                updateLossList(errors)
                t = time.time() - iter_start_time 
                print_current_errors(total_step, errors, t)
            total_step += batch_size

            # Judge done
            if total_step > iteration:
                break
        model.update_learning_rate()

if __name__ == '__main__':
    # Generate data
    dataset = sunnerData.ImageDataset(
        root_list = ['./train2014', './wait'],
        sample_method = sunnerData.OVER_SAMPLING,
        use_cv = False,
        transform = transforms.Compose([
            sunnertransforms.Rescale((160, 320), use_cv = False),
            sunnertransforms.ToTensor(),
            sunnertransforms.ToFloat(),

            # BHWC -> BCHW
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize([127., 127., 127.], [127., 127., 127.])
        ]) 
    )
    loader = sunnerData.ImageLoader(dataset, batch_size=4, shuffle=True, num_workers = 2)
    print('training image = %d' % loader.getImageNumber())

    # --------------------------------------------------------------
    # Train (Usual cycleGAN)
    # --------------------------------------------------------------
    log_name = os.path.join(save_dir, log_name)
    csv_name = os.path.join(save_dir, csv_name)
    """
    model = CycleGAN(save_dir, \
        isTrain = True, \
        input_channel = 3, \
        output_channel = 3, \
        base_filter = 32, \
        batch_size = 4, \
        use_dropout = False, \
        use_gpu = True)
    """
    model = MaskCycleGAN(save_dir, \
        isTrain = True, \
        input_channel = 3, \
        output_channel = 3, \
        base_filter = 32, \
        batch_size = 4, \
        use_dropout = False, \
        use_gpu = True)
    train(model, loader, save_dir)