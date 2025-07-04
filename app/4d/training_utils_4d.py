import os
import glob
import numpy as np
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, backend as k
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from unet3plus import *
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from keras.callbacks import EarlyStopping, ModelCheckpoint
import neptune
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import patches as mpatches
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.colors import to_hex
import tensorflow_mri as tfmri
import tensorflow as tf
import matplotlib.animation as animation
from losses import *
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
import skimage
from volumentations import *
from unet_4d import build_4d_unet
import matplotlib.gridspec as gridspec

import nibabel as nib

channel_dict = {1:'Myocardium',
                2:'Blood Pool'}

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def load_nii(nii_path):
    file = nib.load(nii_path)
    data = file.get_fdata()
    return data


def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val)
    return image



class CustomDataGen():    
    def __init__(self, 
                 patients,
                 cohort,
                 data_path,
                ):
        self.patients = patients
        self.data_path = data_path

    
    def data_generator(self):
        for patient in self.patients:

            image = load_nii(f'{self.data_path}/images/{patient}.nii.gz')
            mask = load_nii(f'{self.data_path}/masks/{patient}.nii.gz').astype('uint8')

            image = np.transpose(image, (3, 2, 0, 1)) # make the depth first dimension
            mask = np.transpose(mask, (3, 2, 0, 1))

            mask = np.eye(3)[mask]

            image = normalize(image)
            image = skimage.exposure.equalize_adapthist(image)
            yield normalize(image[...,np.newaxis]), mask
        
    def get_gen(self):
        return self.data_generator()    


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name, all_patients, run, data_path, output_types,  save_every = 10):
        super(CustomCallback, self).__init__()
        self.save_every = save_every
        self.model_name = model_name
        self.all_patients = all_patients
        self.run = run
        self.data_path = data_path
        self.output_types = output_types
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        self.counter +=1
        if self.counter % self.save_every == 0:
            evaluate(self.model_name, self.all_patients, self.run, self.data_path, self.output_types)

    def on_train_end(self, epoch, logs=None):
        evaluate(self.model_name, self.all_patients, self.run, self.data_path, self.output_types)


def make_video(image, true_mask, pred_mask, dice_val, save_path, time_skip = 3):

    position = image.shape[1]
    timesteps = image.shape[0]

    # Grid layout for each group
    grid_rows = int(np.sqrt(position) + 0.5)
    grid_cols = (position + grid_rows - 1) // grid_rows

    # GridSpec with a narrow gap column in between
    total_cols = grid_cols * 2 + 1  # +1 for spacer
    gap = 0.1
    width_ratios = [1]*grid_cols + [gap] + [1]*grid_cols  # spacer ~4% width of one tile

    fig = plt.figure(figsize=((grid_cols * 2 + gap)*2, grid_rows * 2 * 1.05))
    fig.patch.set_facecolor('black')  # Set figure background to black

    gs = gridspec.GridSpec(grid_rows, total_cols, width_ratios=width_ratios,
                        wspace=0, hspace=0)

    # Prepare axes manually
    axes = np.empty((grid_rows, grid_cols * 2), dtype=object)
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Left grid
            axes[row, col] = fig.add_subplot(gs[row, col])
            axes[row, col].axis('off')
            axes[row, col].patch.set_facecolor('black')  # axes background black

            # Right grid
            axes[row, col + grid_cols] = fig.add_subplot(gs[row, col + grid_cols + 1])
            axes[row, col + grid_cols].axis('off')
            axes[row, col + grid_cols].patch.set_facecolor('black')  # axes background black

    # Add fixed labels in white
    fig.text(0.25, 0.96, 'Ground Truth', ha='center', fontsize='xx-large', color='white')
    fig.text(0.75, 0.96, 'Prediction', ha='center', fontsize='xx-large', color='white')
    fig.text(0.5, 0.96, f'Dice = {dice_val:.2f}', ha='center', fontsize='xx-large', color='white')

    time_skip = 3
    frames = []
    for time in range(0, timesteps, time_skip):
        ttl = plt.text(0.5, 1, f'Timestep = {time + 1}/{timesteps}',
                    ha='center', va='bottom', transform=axes[0, 0].transAxes,
                    fontsize='x-large', color='white')  # timestep text white
        artists = [ttl]

        for row, col in np.ndindex(grid_rows, grid_cols):
            pos = row * grid_cols + col
            if pos < position:
                ax1 = axes[row, col]
                artists.append(ax1.imshow(image[time, pos, :, :], cmap='gray',
                                        vmin=np.min(image), vmax=np.max(image)))
                artists.append(ax1.imshow(true_mask[time, pos, :, :, 1],
                                        alpha=true_mask[time, pos, :, :, 1] * 0.5, cmap='cool_r'))
                artists.append(ax1.imshow(true_mask[time, pos, :, :, 2],
                                        alpha=true_mask[time, pos, :, :, 2] * 0.5, cmap='cool'))

                ax2 = axes[row, col + grid_cols]
                artists.append(ax2.imshow(image[time, pos, :, :], cmap='gray',
                                        vmin=np.min(image), vmax=np.max(image)))
                artists.append(ax2.imshow(pred_mask[time, pos, :, :, 1],
                                        alpha=pred_mask[time, pos, :, :, 1] * 0.5, cmap='cool_r'))
                artists.append(ax2.imshow(pred_mask[time, pos, :, :, 2],
                                        alpha=pred_mask[time, pos, :, :, 2] * 0.5, cmap='cool'))

        frames.append(artists)

    plt.subplots_adjust(left=(gap/position) * 0.5, right= 1 - ((gap/position) * 0.5), top=0.95, bottom=0, wspace=0, hspace=0)
    ani = animation.ArtistAnimation(fig, frames)
    ani.save(f'{save_path}.gif', fps=round(timesteps / time_skip), writer='pillow')
    plt.close()

    


def evaluate(model_name, all_patients, run, data_path, output_types, time_skip = 3):
    cohorts = ['test']
    run[model_name].upload(f'models/{model_name}.h5')
    df = []
    for cohort in cohorts:
        dices = []


        model = build_4d_unet(input_shape=(32, None, 128, 128, 1), num_classes=3)
        model.load_weights(f'models/{model_name}.h5')

        for patient in all_patients[cohort]:
            test_gen = CustomDataGen(patients = [patient], 
                                        cohort = 'test', 
                                        data_path = data_path).get_gen

            test_ds = tf.data.Dataset.from_generator(test_gen, output_types=output_types)
            test_ds = test_ds.batch(1).prefetch(-1)

            pred_mask = model.predict(test_ds)[0]

            pred_mask = get_one_hot(np.argmax(pred_mask,axis = -1), 3)
            

            Path(f'results/{model_name}/masks').mkdir(parents=True, exist_ok=True)
            nib_mask = nib.Nifti1Image(np.argmax(pred_mask, -1), affine=np.eye(4), dtype = 'uint8')
            nib.save(nib_mask, f'results/{model_name}/masks/{patient}.nii.gz')

            image, true_mask = next(iter(test_ds))
            image = np.array(image[0])
            true_mask = np.array(true_mask[0])
            print(true_mask.shape, pred_mask.shape)
            

            for channel in range(1,3):
                dice_val = single_dice(true_mask[...,channel], pred_mask[...,channel])
                df.append({'Patient':patient,
                            'Structure':channel_dict[channel],
                            'Dice':dice_val})

            dice_val = single_dice(true_mask[...,1:], pred_mask[...,1:])

            # plot segmentation video
            make_video(image, true_mask, pred_mask, dice_val, save_path = f'results/{model_name}/{patient}', time_skip = time_skip)
            Path(f'results/{model_name}').mkdir(parents=True, exist_ok=True)
            run[f'results/{cohort}/{patient}'].upload(f'results/{model_name}/{patient}.gif')
            plt.close()
        df = pd.DataFrame.from_records(df)
        df.to_csv(f'segmentation_{model_name}.csv', index = False)
