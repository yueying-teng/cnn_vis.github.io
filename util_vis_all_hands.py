

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math

import skimage
from skimage.transform import resize
import skimage.filters

from vis.utils import utils
from vis.visualization import activation_maximization
from vis.input_modifiers import Jitter

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras import backend as K
from keras import activations

IMG_SIZE = 224


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip image tensor to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x



def path_to_tensor(img_path):
    """
    read image data to the four dimensional tensor format and
    preprocessed according to the requirements of VGG16
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size = (IMG_SIZE, IMG_SIZE))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x



def display_images(images, titles = None, cols = 5, interpolation = None, cmap = "Greys_r"):
    """
    images: A list of images. it can be either:
        - A list of Numpy arrays. Each array represents an image.
        - A list of lists of Numpy arrays. In this case, the images in
          the inner lists are concatentated to make one image.
    """
    titles = titles or [""] * len(images)
    rows = math.ceil(len(images)/ cols)
    height_ratio = 1.2 * (rows/ cols) * (0.5 if type(images[0]) is not np.ndarray else 1)
    plt.figure(figsize = (15, 15 * height_ratio))

    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.axis("off")
        # If image is a list, merge them into one image.
        if type(image) is not np.ndarray:
            image = np.concatenate(image, axis = 1)

        plt.title(title, fontsize = 9)
        plt.imshow(image, cmap = cmap, interpolation = interpolation)
        i += 1
        


# activation visualization

def read_layer(model, test_img_array, layer_name):
    '''
    return activation values for the specified layer
    '''
    get_layer_output = K.function([model.layers[0].input], [model.get_layer(layer_name).output])
    outputs = get_layer_output([test_img_array])[0]
    print(outputs.shape)
    
    # return the plotable image tensor 
    return outputs[0]


def view_layer(model, test_img_array, layer_name, cols = 5):
    outputs = read_layer(model, test_img_array, layer_name)
    display_images([outputs[:, :, i] for i in range(10)], cols = cols)

    


# normalization used in heatmap
def normalize(image):
    """Takes a tensor of 3 dimensions (height, width, colors) and normalizes it's values
    to be between 0 and 1 so it's suitable for displaying as an image."""
    image = image.astype(np.float32)
    
    return (image - image.min()) / (image.max() - image.min() + 1e-5)




def apply_mask(image, mask):
   # Resize mask to match image size
   mask = skimage.transform.resize(normalize(mask), image.shape[: 2])[:,:,np.newaxis].copy()
   # Apply mask to image
   image_heatmap = image * mask
   
   display_images([image_heatmap], cols = 2)

