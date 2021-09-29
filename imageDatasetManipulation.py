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
        plt.show()
        break


for image_batch, labels_batch in train_data:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# images en rvb 0,255 , besoin dêtre mise entre 0 & 1 
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)


normalized_data = train_data.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_data))
first_image = image_batch[0]
# affiche la valeur de l'image entre 0 et 1
print(np.min(first_image), np.max(first_image))


# OPTIMISATION AVEC .CACHE ET .PREFETCH, VOIR FEELINGANALYSIS.PY
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
validate_data = validate_data.cache().prefetch(buffer_size=AUTOTUNE)


# CRÉATION DU MODÈLE

# nb de catégories de fleurs
num_classes = 5
# note sur le mode d'activation: relu = laisse passer les valeurs au dessus de 1, ne pas utiliser en dernière couche
#  sigmoid: proba comprise entre 0 et 1 , très utilisée pour les classification binaires
model = tf.keras.Sequential([
    # 1 ère couche de redimensionnement de la valeur rgb des images comme ci-dessus pour abaisser le nombres de données d'entrées pour le réseau
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    # couches successives de Convolution et de maxpooling voir la doc
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    #  couches successives de Convolution et de maxpooling voir la doc
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    #  couches successives de Convolution et de maxpooling voir la doc
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    # couche qui applatis les pixels
    tf.keras.layers.Flatten(),
    # couche de 128 neurones densément connectés
    tf.keras.layers.Dense(128, activation='relu'),
    # couche finale qui à 5 sorties possibles, 5 catégories de fleurs
    tf.keras.layers.Dense(num_classes)
])

# COMPILATION DU MODELE, OPTIMISATION ET FONCTION DE LOSS ET ACC
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# APPRENTISSAGE DU MODELE
        #   durée des itérations longues en raison du nombre de couches de neurones successives
model.fit(
    # données d'entrée
    train_data,
    # données attendues
    validation_data=validate_data,
    # nb itérations
    epochs=3
)

# affiche le nombre de données comprises dans l'ensemble train_data
print(tf.data.experimental.cardinality(train_data).numpy())








