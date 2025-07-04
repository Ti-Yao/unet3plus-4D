import os
import glob
import numpy as np
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, backend as k
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from keras.callbacks import EarlyStopping, ModelCheckpoint
import neptune
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import patches as mpatches
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.colors import to_hex
# import tensorflow_mri as tfmri
import tensorflow as tf
import matplotlib.animation as animation
from losses import *
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
import skimage
# from volumentations import *
from unet3plus_4D_time import *
import matplotlib.gridspec as gridspec
import albumentations as A
import nibabel as nib
from scipy.stats import truncnorm


import psutil

channel_dict = {1:'Myocardium',
                2:'Blood Pool'}

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def load_nii(nii_path):
    file = nib.load(nii_path)
    data = file.get_fdata()
    return data

def find_crop_box(mask, crop_factor):
    '''
    Calculated a bounding box that contains the masks inside.

    Parameters:
    mask: np.array
        A binary mask array, which should be the flattened 3D multislice mask, where the pixels in the z-dimension are summed
    crop_factor: float
        A scaling factor for the bounding box
    Returns:
    list
        A list containing the coordinates of the bounding box [x_min, y_min, x_max, y_max]. These co-ordinates can be used to crop each slice of the input multislice image.
    '''
    # Check shape of the input is 2D
    if len(mask.shape) != 2:
        raise ValueError("Input mask must be a 2D array")
    
    y = np.sum(mask, axis=1) # sum the masks across columns of array, returns a 1D array of row totals
    x = np.sum(mask, axis=0) # sum the masks across rows of array, returns a 1D array of column totals

    top = np.min(np.nonzero(y)) - 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. top extent of mask)
    bottom = np.max(np.nonzero(y)) + 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the maximum value and adds 1 (i.e. bottom extent of mask)

    left = np.min(np.nonzero(x)) - 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. left extent of mask)
    right = np.max(np.nonzero(x)) + 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the maximum value and adds 1 (i.e. right extent of mask)
    if abs(right - left) > abs(top - bottom):
        largest_side = abs(right - left) # Find the largest side of the bounding box
    else:
        largest_side = abs(top - bottom)
    x_mid = round((left + right) / 2) # Find the mid-point of the x-length of mask
    y_mid = round((top + bottom) / 2) # Find the mid-point of the y-length of mask
    half_largest_side = round(largest_side * crop_factor / 2) # Find half the largest side of the bounding box (crop factor scales the largest side to ensure whole heart and some surrounding is captured)
    x_max, x_min = round(x_mid + half_largest_side), round(x_mid - half_largest_side) # Find the maximum and minimum x-values of the bounding box
    y_max, y_min = round(y_mid + half_largest_side), round(y_mid - half_largest_side) # Find the maximum and minimum y-values of the bounding box
    if x_min < 0:
        x_max -= x_min # if x_min less than zero, expand the x_max value by the absolute value of x_min, to ensure bounding box is same size
        x_min = 0

    if y_min < 0:
        y_max -= y_min # if y_min less than zero, expand the y_max value by the absolute value of y_min, to ensure bounding box is same size
        y_min = 0

    return [x_min, y_min, x_max, y_max]

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val)
    return image

def clip_outliers(img, lower_percentile=1, upper_percentile=99):
    lower = np.percentile(img, lower_percentile)
    upper = np.percentile(img, upper_percentile)
    clipped_img = np.clip(img, lower, upper)
    return clipped_img

def transpose_channels(image, input_channel_order, target_channel_order):
    """
    Given input and target channel orders, return the transpose order.

    Args:
        input_order (list or str): Current order of axes, e.g., ['H','W','D','T','C']
        target_order (list or str): Desired order of axes, e.g., ['D','T','H','W','C']

    Returns:
        list: Indices to use in np.transpose to convert input_order to target_order
    """
    return np.transpose(image, [input_channel_order.index(axis) for axis in target_channel_order])

def truncated_normal_sample(low=1.2, high=1.8, mean=1.5, std=0.2):
    a, b = (low - mean) / std, (high - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std)

def apply_transform(image, mask, transform_fn, input_channel_order):
    image = transpose_channels(image, input_channel_order, ['T','H', 'W','D'])
    mask = transpose_channels(mask, input_channel_order, ['T','H', 'W','D'])

    image_list = list(image)
    mask_list = list(mask)

    aug = transform_fn(images=image_list, masks=mask_list)
    aug_image = np.stack(aug['images'],0)   
    aug_mask = np.stack(aug['masks'],0)

    return aug_image, aug_mask

def crop_image_mask(image, mask, cohort, crop_factor, translation_factor, input_channel_order):
    """
    Crop the image and mask based on the bounding box of the mask,
    with optional scaling and translation.

    Args:
        image (np.ndarray): Input image to crop.
        mask (np.ndarray): Binary mask to determine crop area.
        crop_factor (float): Factor to scale the bounding box size.
        translation_factor (float): Fractional factor to translate the crop box (as a proportion of box size).
        input_channel_order (list): Original channel order of the input arrays.

    Returns:
        tuple: Cropped image and mask.
    """
    # Convert to standardized channel order
    image = transpose_channels(image, input_channel_order, ['T', 'D', 'H', 'W'])
    mask = transpose_channels(mask, input_channel_order, ['T', 'D', 'H', 'W'])

    # Collapse time and depth axes to get 2D projection for crop region
    sum_mask = np.max(mask, axis=(0, 1))

    if cohort == 'train':
        crop_factor = truncated_normal_sample(low=crop_factor - 0.3, high=crop_factor + 0.3, mean=crop_factor, std=0.5)
    else:
        crop_factor = 1.5
    # Find initial crop box
    x_min, y_min, x_max, y_max = find_crop_box(sum_mask, crop_factor=crop_factor)

    # Compute box size
    box_width = x_max - x_min
    box_height = y_max - y_min

    if cohort == 'train':
        # Random shift in both directions
        x_shift = int(np.random.uniform(-translation_factor, translation_factor) * box_width)
        y_shift = int(np.random.uniform(-translation_factor, translation_factor) * box_height)

    else:
        x_shift = 0
        y_shift = 0

    # Translate crop box
    x_min = max(0, x_min + x_shift)
    y_min = max(0, y_min + y_shift)
    x_max = min(sum_mask.shape[1], x_max + x_shift)
    y_max = min(sum_mask.shape[0], y_max + y_shift)

    # Crop image and mask
    image = image[:, :, y_min:y_max, x_min:x_max]
    mask = mask[:, :, y_min:y_max, x_min:x_max]

    return image, mask


class CustomDataGen():    
    def __init__(self, 
                 patients,
                 cohort,
                 data_path,
                ):
        self.patients = patients
        self.data_path = data_path
        self.cohort = cohort

    
    def data_generator(self):
        pre_transform = A.Compose(
            [
                A.Affine(rotate=[-30,30],  p=0.8),
                A.RandomRotate90(p = 0.2),
            ])

        post_transform = A.Compose([
            A.Normalize(normalization='min_max',max_pixel_value= 1.0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=0.6),
            A.ToFloat(),
            A.GaussNoise(std_range=[0, 0.03],per_channel=True, p = 0.3),
            A.Downscale(scale_range=[0.5, 1], p = 0.5),
            A.Resize(height=128,width=128, p = 1
        )

        ])

        resize_transform = A.Compose([
            A.Normalize(normalization='min_max',max_pixel_value= 1.0, p=1.0),
            A.Resize(height=128,width=128, p = 1
        )

        ])
        
        for patient in self.patients:
            try:
                image = load_nii(f'{self.data_path}/images/{patient}.nii.gz')
                mask = load_nii(f'{self.data_path}/masks/{patient}.nii.gz').astype('uint8')

                if image.shape[2] == mask.shape[2]:
                    
                    # if self.cohort == 'train': # bio-volumentations
                    #     image, mask = crop_image_mask(image, mask, cohort = self.cohort, crop_factor=1.5, translation_factor = 0.1, input_channel_order = ['H','W','D','T'])
                    #     image, mask = apply_transform(image, mask, post_transform, input_channel_order = ['T','D','H','W'])

                    if self.cohort == 'train': # albumentations
                        image, mask = apply_transform(image, mask, pre_transform, input_channel_order = ['H','W','D','T'])
                        image, mask = crop_image_mask(image, mask, cohort = self.cohort, crop_factor=1.5, translation_factor = 0.1, input_channel_order = ['T','H','W','D'])
                        image, mask = apply_transform(image, mask, post_transform, input_channel_order = ['T','D','H','W'])

                    else:
                        image, mask = crop_image_mask(image, mask, cohort = self.cohort, crop_factor=1.5, translation_factor = 0, input_channel_order = ['H','W','D','T'])
                        image, mask = apply_transform(image, mask, resize_transform, input_channel_order = ['T','D','H','W'])

                    image = transpose_channels(image, ['T','H','W','D'], ['T','D','H','W'])
                    mask = transpose_channels(mask, ['T','H','W','D'], ['T','D','H','W'])

                    mask = np.eye(3)[mask.astype(np.uint8)]

                    image = clip_outliers(image, lower_percentile=1, upper_percentile=99)
                    image = normalize(image)
                    image = skimage.exposure.equalize_adapthist(image)
                    yield normalize(image[...,np.newaxis]).astype('float32'), mask.astype('uint8')
                else:
                    del image, mask
            except Exception as e:
                print(patient, e)
            
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


def make_video(image, true_mask, pred_mask, dice_vals, save_path, time_skip = 3):

    position = image.shape[1]
    timesteps = image.shape[0]

    # Grid layout for each group
    grid_rows = int(np.sqrt(position) + 0.5)
    grid_cols = (position + grid_rows - 1) // grid_rows

    # GridSpec with a narrow gap column in between
    total_cols = grid_cols * 2 + 1  # +1 for spacer
    gap = 0.1
    width_ratios = [1]*grid_cols + [gap] + [1]*grid_cols  # spacer ~4% width of one tile

    fig = plt.figure(figsize=((grid_cols * 2 + gap)*1.2, grid_rows * 1.2 * 1.05))
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
    fig.text(0.25, 0.96, 'Ground Truth', ha='center', fontsize='large', color='white')
    fig.text(0.75, 0.96, 'Prediction', ha='center', fontsize='large', color='white')
    fig.text(0.4, 0.96, f'Myo = {dice_vals[0]:.2f}', ha='center', fontsize='large', color='#01FEFF')
    fig.text(0.6, 0.96, f'Endo = {dice_vals[1]:.2f}', ha='center', fontsize='large', color='#FC02FF')

    time_skip = 3
    frames = []
    for time in range(0, timesteps, time_skip):
        ttl = plt.text(0.5, 1, f'Timestep = {time + 1}/{timesteps}',
                    ha='center', va='bottom', transform=axes[0, 0].transAxes, color='white')  # timestep text white
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

    


def evaluate(model_name, all_patients, run, data_path, output_signature, time_skip = 3):
        cohorts = ['test']
        run[model_name].upload(f'models/{model_name}.h5')
        df = []
        for cohort in cohorts:
            dices = []


            model = build_unet3plus_4D(input_shape=(32, None, 128, 128, 1), num_classes=3)
            model.load_weights(f'models/{model_name}.h5')

            for patient in all_patients[cohort]:
                try:
                    test_gen = CustomDataGen(patients = [patient], 
                                                cohort = 'test', 
                                                data_path = data_path).get_gen

                    test_ds = tf.data.Dataset.from_generator(test_gen, output_signature=output_signature)
                    test_ds = test_ds.batch(1).prefetch(-1)

                    pred_mask = model.predict(test_ds)
                    if isinstance(pred_mask, list):
                        pred_mask = pred_mask[-1]  # If model returns a list, take the
                    
                    pred_mask = pred_mask[0]
                    pred_mask = get_one_hot(np.argmax(pred_mask,axis = -1), 3)
                    

                    Path(f'results/{model_name}/masks').mkdir(parents=True, exist_ok=True)
                    nib_mask = nib.Nifti1Image(np.argmax(pred_mask, -1), affine=np.eye(4), dtype = 'uint8')
                    nib.save(nib_mask, f'results/{model_name}/masks/{patient}.nii.gz')

                    image, true_mask = next(iter(test_ds))
                    image = np.array(image[0])
                    true_mask = np.array(true_mask[0])
                    print(true_mask.shape, pred_mask.shape)
                    
                    dice_vals = []

                    for channel in range(1,3):
                        dice_val = single_dice(true_mask[...,channel], pred_mask[...,channel])
                        df.append({'Patient':patient,
                                    'Structure':channel_dict[channel],
                                    'Dice':dice_val})
                        dice_vals.append(dice_val)

                    # plot segmentation video
                    make_video(image, true_mask, pred_mask, dice_vals, save_path = f'results/{model_name}/{patient}', time_skip = time_skip)
                    Path(f'results/{model_name}').mkdir(parents=True, exist_ok=True)
                    run[f'results/{cohort}/{patient}'].upload(f'results/{model_name}/{patient}.gif')
                    plt.close()

                except Exception as e:
                    print(e, patient)   
            df = pd.DataFrame.from_records(df)
            df.to_csv(f'segmentation_{model_name}.csv', index = False)

            for channel in range(1,3):
                structure = channel_dict[channel]
                run[f'{structure}_dice_median'] = df.loc[(df['Structure'] == structure)]['Dice'].median()
                run[f'{structure}_dice_iqr25'] = df.loc[(df['Structure'] == structure)]['Dice'].quantile(0.25)
                run[f'{structure}_dice_iqr75'] = df.loc[(df['Structure'] == structure)]['Dice'].quantile(0.75)
 

        
class MemoryCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_usage_mb = mem_info.rss / 1024 / 1024  # Convert bytes to MB
        print(f"Epoch {epoch+1}: System memory usage: {mem_usage_mb:.2f} MB")
        tf.keras.backend.clear_session()
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_usage_mb = mem_info.rss / 1024 / 1024  # Convert bytes to MB
        print(f"Epoch {epoch+1}: System memory usage after clear: {mem_usage_mb:.2f} MB")
        print("End epoch, collect garbage")

