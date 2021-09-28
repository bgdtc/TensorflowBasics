#CLASSIFICATION DE TEXTE

# IMPORT MODULES
from typing import Text
import numpy
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
from numpy.lib.function_base import vectorize
import tensorflow as tf

from tensorflow.keras import layers 
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization




# lien vers le dataset d'avis de films
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# variable dataset
dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                   untar=True, cache_dir='.',
                                   cache_subdir='')
# folder dataset
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

# affiche les folders dans le folder du dataset
print(os.listdir(dataset_dir))

# folder du train data
train_dir = os.path.join(dataset_dir, 'train')
# on affiche le contenu du folder train
print(os.listdir(train_dir))

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)



# DÉFINITIONS DU RÔLES DES DONNÉES

batch_size = 32
seed = 48

# ensemble d'entrainement
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

# ensemble de validation
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)

# ensemble  de test
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size
)



# SUPPRESSION DES BALISES HTML DU TEXTE ET AUTRES SALOPERIES

# lowercase-isation + dégage le html
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')
# vectorization, tokenisation des data, en bref ça assigne les data à des tokens & nombres

max_features = 10000
sequence_length = 250

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)


# rendre le dataset uniquement en texte ?

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# afficher le résultat de notre préprocessing du dataset

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("review 0: ", first_review)
print("label 0: ", raw_train_ds.class_names[first_label])
print("review vectorisée: ", vectorize_text(first_review, first_label))


# tokens correspondent à chaine de caractère
# j'affiche le token 777 qui correspond à " tom "
print("777 -> ", vectorize_layer.get_vocabulary()[777])



#  on applique ce préprocessing à l'ensemble du dataset pour la suite

# au jeu de données de train
train_ds = raw_train_ds.map(vectorize_text)
# au jeu de données de validation
val_ds = raw_val_ds.map(vectorize_text)
# et au jeu de données de test
test_ds = raw_test_ds.map(vectorize_text)


#  utilisation de .cache et .prefetch pour que les entrées et sorties ne soit pas bloquantes
# .cache garde les data en memoire apres chargement, evite trop de data a charger d'un coup
# .prefetch traite les data pendant l'execution de l'entrainement du modèle 


AUTOTUNE = tf.data.AUTOTUNE

# .cache & .prefetch sur les 3 ensembles de data

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) 
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)



# CRÉATION DU RÉSEAU DE NEURONES

embedding_dim = 16


model = tf.keras.Sequential([
    # première couche de neurones : cherche vecteur d'integration pour chaque index de mot
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    # couche globalaveragepooling1d gère les data d'entrée qui ont une longeur variable 'bite', 'couille'
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    # dernière couche, un seul neurone de sortie
    layers.Dense(1)
])

# affiche la configuration du modèle
print(model.summary())

# COMPILATION DU MODÈLE, FONCTION DE PERTE ETC... 

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# ENTRAINEMENT DU MODÈLE

# nombre d'itérations  ~15 correct pas de sur ou sous apprentissage
epochs = 15

history = model.fit(
    # données d'entrainement
    train_ds,
    # données attendues
    validation_data=val_ds,
    # itérations
    epochs=epochs
)


# résultats de l'entrainement sur les data de test

loss, accuracy = model.evaluate(test_ds)

# affiche la précision et la perte  ~86% pour 20 itérations
print("loss->",loss)
print("précision->", accuracy)

# afficher ces résultats sur un graphique

# affiche les 4 métriques qui surveillent le modèle pendant l'entrainement
history_dict = history.history
print(history_dict.keys())


#  on simplifie le bordel et on affiche ça dans un graphique

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)


# GRAPHIQUE POUR LA LOSS
# dans matplotlib, bo = blue dot , b = solid blue line
plt.plot(epochs, loss, 'bo', label='ENTRAINEMENT LOSS')
plt.plot(epochs, val_loss, 'b', label='VALIDATION LOSS')
plt.title(' LOSS POUR L\'ENTRAINEMENT ET LA VALIDATION ')
plt.xlabel('Itérations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# GRAPHIQUE POUR LA PRÉCISION
plt.plot(epochs, acc, 'bo', label='PRÉCISION ENTRAINEMENT')
plt.plot(epochs, val_acc, 'b', label='PRÉCISION VALIDATION')
plt.title('PRÉCISION POUR LE TRAIN ET LA VALIDATION')
plt.xlabel('Itérations')
plt.ylabel('Précision')
plt.legend(loc='lower right')
plt.show()


#  EXPORT DU MODÈLE ENTRAINÉ ! 

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

# compilation
export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)
# affichage loss/accuracy
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)


# PRÉDICTIONS SUR DES DONNÉES A LA MANO

examples = [
    "This film was the best i ever seen !",
    "The film is normal.",
    "This movie is shitty"
]

# affichage des prédictions
print(export_model.predict(examples))

# ~1 = Avis positif, joyeux
# ~0 = Avis négatif, malheureux




