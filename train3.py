import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Add, UpSampling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def build_generator(input_shape):
    def residual_block(x):
        res = Conv2D(64, (3, 3), padding='same')(x)
        res = BatchNormalization()(res)
        res = LeakyReLU(alpha=0.2)(res)
        res = Conv2D(64, (3, 3), padding='same')(res)
        res = BatchNormalization()(res)
        res = Add()([res, x])
        return res

    inputs = Input(shape=input_shape)
    x = Conv2D(64, (9, 9), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    res = x
    for _ in range(16):
        res = residual_block(res)

    x = Add()([x, res])

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = UpSampling2D(size=2)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = UpSampling2D(size=2)(x)
    x = LeakyReLU(alpha=0.2)(x)

    outputs = Conv2D(3, (9, 9), padding='same', activation='tanh')(x)

    return Model(inputs, outputs)

def build_discriminator(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    filters = 64
    for i in range(3):
        x = Conv2D(filters, (3, 3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        filters *= 2

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs)

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(None, None, 3))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    return Model(gan_input, gan_output)


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Funzione per caricare e preprocessare il dataset
def load_images(corrupted_dir, high_quality_dir, img_size):
    corrupted_images = []
    high_quality_images = []

    for img_name in os.listdir(corrupted_dir):
        try:
            corrupted_img = load_img(os.path.join(corrupted_dir, img_name), target_size=(img_size, img_size))
            high_quality_img = load_img(os.path.join(high_quality_dir, img_name), target_size=(img_size*4, img_size*4))

            corrupted_images.append(img_to_array(corrupted_img))
            high_quality_images.append(img_to_array(high_quality_img))
        except Exception as e:
            print(f"Errore nel caricamento dell'immagine {img_name}: {e}")

    return np.array(corrupted_images), np.array(high_quality_images)

# Dimensione delle immagini
img_size = 64

# Carica le immagini (aggiorna i percorsi con le tue directory)
corrupted_dir = 'dataset/low_quality'
high_quality_dir = 'dataset/high_quality'
corrupted_images, high_quality_images = load_images(corrupted_dir, high_quality_dir, img_size)

# Normalizzazione delle immagini
corrupted_images = corrupted_images / 127.5 - 1
high_quality_images = high_quality_images / 127.5 - 1

# Costruisci i modelli
generator = build_generator((img_size, img_size, 3))
discriminator = build_discriminator((img_size * 4, img_size * 4, 3))
gan = build_gan(generator, discriminator)

# Compila i modelli
generator.compile(optimizer='adam', loss='mse')
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Addestramento
epochs = 10000
batch_size = 16
for epoch in range(epochs):
    for i in range(0, len(corrupted_images), batch_size):
        corrupted_batch = corrupted_images[i:i + batch_size]
        high_quality_batch = high_quality_images[i:i + batch_size]

        generated_images = generator.predict(corrupted_batch)

        real_labels = np.ones((len(high_quality_batch), 1))
        fake_labels = np.zeros((len(generated_images), 1))

        d_loss_real = discriminator.train_on_batch(high_quality_batch, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)

        g_loss = gan.train_on_batch(corrupted_batch, real_labels)

    print(f"Epoch {epoch+1}/{epochs} - D Loss Real: {d_loss_real:.4f} - D Loss Fake: {d_loss_fake:.4f} - G Loss: {g_loss:.4f}")

# Salva il modello generatore
generator.save('srgan_generator.h5')




import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt

# Carica il modello generatore addestrato
generator = tf.keras.models.load_model('srgan_generator.h5')

# Funzione per correggere un'immagine corrotta
def correct_image(corrupted_image_path, generator, img_size):
    img = load_img(corrupted_image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 127.5 - 1
    img_array = np.expand_dims(img_array, axis=0)

    corrected_img_array = generator.predict(img_array)
    corrected_img = np.squeeze(corrected_img_array) * 127.5 + 127.5
    corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)

    return array_to_img(corrected_img)

# Correggi un'immagine esempio e visualizza il risultato
example_img_path = 'dataset/low_quality/landascape_0.jpeg'
corrected_img = correct_image(example_img_path, generator, img_size)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Corrupted Image')
plt.imshow(load_img(example_img_path, target_size=(img_size, img_size)))

plt.subplot(1, 2, 2)
plt.title('Corrected Image')
plt.imshow(corrected_img)
plt.show()
