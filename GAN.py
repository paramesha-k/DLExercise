import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# Load the MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images.astype('float32') - 127.5) / 127.5
train_images = train_images.reshape(train_images.shape[0], 784)
train_images = train_images[np.random.randint(train_images.shape[0], size=10)]

def create_generator():
    generator = Sequential()
    generator.add(Dense(256, input_dim=100))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization())
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization())
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization())
    generator.add(Dense(784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return generator

def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return discriminator

def create_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def train_gan(gan, generator, discriminator, epochs=1, batch_size=128):
    batch_count = train_images.shape[0] // batch_size

    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        for _ in range(batch_count):
            noise = np.random.normal(0, 1, size=[batch_size, 100])
            generated_images = generator.predict(noise)
            image_batch = train_images[np.random.randint(0, train_images.shape[0], size=batch_size)]

            X = np.concatenate([image_batch, generated_images])

            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)

            noise = np.random.normal(0, 1, size=[batch_size, 100])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)

      
    plot_generated_images(generator, e+1)

def plot_generated_images(generator, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.close()

if __name__ == "__main__":
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(generator, discriminator)

    # Training the GAN
    train_gan(gan, generator, discriminator, epochs=1000, batch_size=10)

