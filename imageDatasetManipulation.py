# IMPORT MODULES

import numpy as np
import os 
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pathlib




# URL DU DATASET
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# IMPORT DU DATASET
data_dir = tf.keras.utils.get_file("flower_photos",
                                    dataset_url,
                                    cache_dir='.',
                                    cache_subdir='',
                                   untar=True)
# OU
dataset_dir = os.path.join(os.path.dirname(data_dir), 'flower_photos')

# CRÉATION DU JEU DE DONNÉES A PARTIR DU DATASET DE FLEURS

# packs de 32 images, de meme taille 
batch_size = 32
img_height = 180
img_width = 180


# JEU D'ENTRAINEMENT
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

#  JEU DE VALIDATIONS

validate_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# affiche catégories de fleurs
categories_fleurs = train_data.class_names
print(categories_fleurs)


# MONTRE LES PHOTOS BATARD


plt.figure(figsize=(10,10))
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(categories_fleurs[labels[i]])
        plt.axis("off")
 





