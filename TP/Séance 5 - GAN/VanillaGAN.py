import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm

class VanillaGAN:
    def __init__(self, discriminator, generator, random_dimension):
        self.discriminator = discriminator
        self.generator = generator
        self.random_dimension = random_dimension
        self.gan = self.build_gan()



    def build_gan(self):
        self.discriminator.trainable = False
        gan_input = keras.Input(shape=(self.random_dimension,))
        gan_output = self.discriminator(self.generator(gan_input))
        gan = keras.models.Model(gan_input, gan_output)
        return gan

    def compile(self, **kwargs):
        self.gan.compile(**kwargs) 


    def generate_images(self, sample, verbose=0):
        noise = np.random.normal(0, 1, size=(sample, self.random_dimension))
        images = self.generator.predict(noise, verbose=verbose)
        return images



    def train(self, X_train, epochs, batch_size, k=1):

        def train_discriminator(real, fake):
            self.discriminator.trainable = True
            discriminator_loss_real = self.discriminator.train_on_batch(real, np.ones((batch_size, 1)))
            discriminator_loss_fake = self.discriminator.train_on_batch(fake, np.zeros((batch_size, 1)))
            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
            return discriminator_loss


        def train_generator():
            self.discriminator.trainable = False
            noise = np.random.normal(0, 1, (batch_size, self.random_dimension))
            generator_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))
            return generator_loss

        

        discriminator_loss_list = []
        generator_loss_list = []

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            for _ in tqdm(range(len(X_train) // batch_size)):
                fake_images = self.generate_images(sample=batch_size)
                real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

                for _ in range(k): discriminator_loss = train_discriminator(real=real_images, fake=fake_images)
                generator_loss = train_generator()

            discriminator_loss_list.append(discriminator_loss)
            generator_loss_list.append(generator_loss)
            performance = self.evaluate_discriminator(X_test=X_train, sample=100)[1]
            print(discriminator_loss)
            print(f"Discriminator loss: {float(discriminator_loss[0]):.4f}, accuracy: {performance:.2f}%")
            print(f"Generator loss: {float(generator_loss[0]):.4f}")

            self.discriminator.save('discriminator_model.keras')
            self.generator.save('generator_model.keras')

        return discriminator_loss_list, generator_loss_list
    


    def evaluate_discriminator(self, X_test, sample, verbose=0):
        noise = np.random.normal(0, 1, (sample, self.random_dimension))
        fake = self.generator.predict(noise, verbose=0)
        real = X_test[np.random.randint(0, X_test.shape[0], sample)]
        X = np.vstack((real, fake))
        y = np.vstack((np.ones((sample, 1)), np.zeros((sample, 1))))
        return self.discriminator.evaluate(X, y, verbose=verbose)
