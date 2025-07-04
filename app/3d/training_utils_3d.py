import os, glob, random, neptune, skimage, json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, backend as k
from keras.callbacks import EarlyStopping, ModelCheckpoint
from unet3plus import *
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
from matplotlib.colors import ListedColormap, to_hex
import tensorflow_mri as tfmri
import pandas as pd
# import albumentations as A
import nibabel as nib
import matplotlib.animation as animation
from pathlib import Path
from layer_util import *
from losses import *
import glob

channel_dict = {1:'Myocardium',
                2:'Blood Pool',}

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
                 data_path,
                 cohort,
                 timesteps = [0, 1]
                ):
        self.patients = patients
        self.cohort = cohort
        self.timesteps = timesteps
        self.data_path = data_path
        
    def data_generator(self):
        for patient in self.patients:
            images = load_nii(f'{self.data_path}/images/{patient}.nii.gz')
            masks = load_nii(f'{self.data_path}/masks/{patient}.nii.gz').astype('uint8')

            for time in self.timesteps:
                image = images[...,time]
                mask = masks[...,time]
                mask = np.eye(3)[mask]

                image = normalize(image)
                image = skimage.exposure.equalize_adapthist(image)
                yield normalize(image[...,np.newaxis]), mask

    def get_gen(self):
        return self.data_generator()    



class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name, all_patients, run, data_path, output_signature,  save_every = 10):
        super(CustomCallback, self).__init__()
        self.save_every = save_every
        self.model_name = model_name
        self.all_patients = all_patients
        self.run = run
        self.data_path = data_path
        self.output_signature = output_signature
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        self.counter +=1
        if self.counter % self.save_every == 0:
            evaluate(self.model_name, self.all_patients, self.run, self.data_path, self.output_signature)

    def on_train_end(self, epoch, logs=None):
        evaluate(self.model_name, self.all_patients, self.run, self.data_path, self.output_signature)

    
def make_video(image, true_mask, pred_mask, dice_vals, save_path):
    fig, axs = plt.subplots(1,2, figsize = (10,5))
    frames = []

    fig.suptitle(f"Myo Dice = {round(dice_vals[0],2)}\nEndo Dice = {round(dice_vals[1],2)}")

    for i in range(0,image.shape[2]):
        p1 = axs[0].imshow(image[...,i, 0],cmap = 'gray', vmin = np.min(image),vmax = np.max(image))
        p2 = axs[1].imshow(image[...,i, 0],cmap = 'gray', vmin = np.min(image),vmax = np.max(image))
        axs[0].set_title('Ground Truth')
        axs[1].set_title('Prediction')

        text = plt.figtext(0.5, 0.05, f"Frame = {i}", ha = 'center', va = 'bottom', fontsize = 12)

        artists = [p1,p2, text]
        for channel in range(1, pred_mask.shape[-1]):
            cmaps = ['gray', 'Blues','jet']
            artists.append(axs[0].imshow(true_mask[...,i,channel],alpha = true_mask[...,i,channel] * 0.5, cmap = cmaps[channel]))
            artists.append(axs[1].imshow(pred_mask[...,i,channel],alpha = pred_mask[...,i,channel] * 0.5, cmap = cmaps[channel]))
        axs[0].axis('off')
        axs[1].axis('off')
        frames.append(artists)

    ani = animation.ArtistAnimation(fig, frames)
    ani.save(f'{save_path}.gif', fps=image.shape[2]/10)
    plt.close()    
    


def evaluate(model_name, all_patients, run, data_path, output_signature):
    cohorts = ['test']
    run[model_name].upload(f'models/{model_name}.h5')
    df = []
    for cohort in cohorts:
        dices = []

        model = tf.keras.models.load_model(f'models/{model_name}.h5', compile = False, custom_objects = {'ResizeAndConcatenate':ResizeAndConcatenate})    

        for patient in all_patients[cohort]:
            for time in [0,1]:
                # try:
                frame  = 'Diastole' if time == 0 else 'Systole'
                print(patient)

                test_gen = CustomDataGen(patients = [patient], 
                                        cohort = 'test',
                                        data_path = data_path,
                                        timesteps=[time]).get_gen

                test_ds = tf.data.Dataset.from_generator(test_gen, output_signature=output_signature)
                test_ds = test_ds.batch(1).prefetch(-1)

                pred_mask = model.predict(test_ds)[-1][0]

                pred_mask = get_one_hot(np.argmax(pred_mask,axis = -1), 3)

                image, true_mask = next(iter(test_ds))
                image = np.array(image[0])
                true_mask = np.array(true_mask[0])
                
                Path(f'results/{model_name}/masks').mkdir(parents=True, exist_ok=True)
                np.save(f'results/{model_name}/masks/{patient}_{frame}.npy', pred_mask)

                dice_vals = []
                
                for channel in channel_dict.keys():
                    dice_val = single_dice(true_mask[...,channel], pred_mask[...,channel])
                    df.append({'Structure':channel_dict[channel],
                                'Dice':dice_val,
                                'Frame':frame})
                    dice_vals.append(dice_val)
                    dices.append(dice_val)

                # plot segmentation video
                make_video(image, true_mask, pred_mask, dice_vals, save_path = f'results/{model_name}/{patient}_{frame}')
                Path(f'results/{model_name}').mkdir(parents=True, exist_ok=True)
                run[f'results/{cohort}/{patient}_{frame}'].upload(f'results/{model_name}/{patient}_{frame}.gif')
                plt.close()
                # except:
                #     pass
                
        try:
            run['median_dice'] = np.median(dices)
            run['iqr25_dice'] = np.quantile(dices, 0.25)    
            run['iqr75_dice'] = np.quantile(dices, 0.75)
            
            df = pd.DataFrame.from_records(df)
            df.to_csv(f'segmentation_{model_name}.csv', index = False)

            for structure in channel_dict.values():
                for frame in ['Diastole' ,'Systole']:
                    run[f'{frame}_{structure}_dice_median'] = df.loc[(df['Structure'] == structure) & (df['Frame'] == frame)]['Dice'].median()
                    run[f'{frame}_{structure}_dice_iqr25'] = df.loc[(df['Structure'] == structure) & (df['Frame'] == frame)]['Dice'].quantile(0.25)
                    run[f'{frame}_{structure}_dice_iqr75'] = df.loc[(df['Structure'] == structure) & (df['Frame'] == frame)]['Dice'].quantile(0.75)
                    
        except:
            pass