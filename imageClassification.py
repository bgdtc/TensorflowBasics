# CLASSIFICATION D'IMAGES 


# IMPORT MODULES 

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

# VERSION TF
print("version tensorflow: ", tf.__version__)

# STOCKAGE DATASET VARIABLE
clothes = tf.keras.datasets.fashion_mnist

# TRI DES CATEGORIES DE DATA DU DATASET
(train_images, train_labels), (test_images, test_labels) = clothes.load_data()

# TYPES DE VETEMENTS DU DATASET
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# MISE À L'ECHELLE ENTRE 0 ET 1 DES IMAGES
train_images = train_images / 255.0
test_images  = test_images / 255.0


# AFFICHAGE DES 25 PREMIÈRES IMAGES DU DATASET (TRAIN DATA)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# CRÉATION DU MODÈLE, AVEC UNE COUCHE QUI PASSE LE TABLEAU 2D DU DATASET EN UNIDIMENTIONNEL
#  UNE COUCHE DE 128 NEURONES INTERCONNECTÉS
#  UNE COUCHE QUI RENVOIE UN TABLEAU LOGITS QUI VA DE 1 A 10 POUR LES 9 CATÉGORIES DIFFÉRENTES DE FRINGUES
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

# ON COMPILE LE MODÈLE AVEC UNE FONCTION QUI NOUS INDIQUE LA PRÉCISION DU MODÈLE
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ON ENTRAINE LE MODÈLE SUR LES IMAGES D'ENTRAINEMENT
model.fit(train_images, train_labels, epochs=35)

# ON AFFICHE LA PRÉCISION DU MODÈLE
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#  ON EFFECTUE UNE PRÉDICTION DE LA CATÉGORIE POUR UNE IMAGE
probability_model = tf.keras.Sequential([model,
                                              tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[4])
print(np.argmax(predictions[4]))
print(test_labels[4])

# ON AFFICHE CETTE PREDICTION DANS UN GRAPHIQUE 
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)
# ON AFFICHE LES PRÉDICTIONS POUR LES 25 IMAGES    
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 4 
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()


num_rows = 5
num_cols = 3 
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()



# ON UTILISE LE MODÈLE ENTRAINÉ POUR PRÉDIRE LA BONNE CATÉGORIE POUR UNE IMAGE
img = test_images[1]

print(img.shape)

img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = probability_model.predict(img)
print(predictions_single)

#  ON RENVOIE LA PRÉDICTION DANS UN TABLEAU 
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

print(np.argmax(predictions_single[0]))

