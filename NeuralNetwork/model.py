import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2DTranspose, Conv3D, Flatten, BatchNormalization, Activation, LeakyReLU, Reshape, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal



class MuseGAN(Model):
    def __init__(self,
                 noise_length=32,
                 discriminator_steps=5,
                 gradient_penalty_weight=10,
                 count_bars=9,
                 count_tracks=4,
                 count_steps_per_bar=16,
                 count_notes=75,
                 is_load=False,
                 trainable=True,
                 dtype='float32'):
        super(MuseGAN, self).__init__()
        self.initializer = RandomNormal(mean=0.0, stddev=0.02)
        self.noise_length = noise_length
        self.discriminator_steps = discriminator_steps
        self.gradient_penalty_weight = gradient_penalty_weight
        self.count_bars = count_bars
        self.count_tracks = count_tracks
        self.count_steps_per_bar = count_steps_per_bar
        self.count_notes = count_notes
        self.discriminator = self.discriminator_initialize()
        self.generator = self.generator_initialize()
        self.is_load = is_load

    def compile(self, discriminator_learning_rate, generator_learning_rate, adam_beta1, adam_beta2):
        #if (self.is_load):
        #    self.load_weights("./weights/checkpoint.weights.h5")

        super(MuseGAN, self).compile()
        self.discriminator_optimizer = Adam(discriminator_learning_rate, adam_beta1, adam_beta2)
        self.generator_optimizer = Adam(generator_learning_rate, adam_beta1, adam_beta2)
        self.discriminator_loss_metric = Mean(name="discriminator_loss")
        self.generator_loss_metric = Mean(name="generator_loss")

    def get_config(self):
        return {
            'initializer': self.initializer,
            'noise_length': self.noise_length,
            'discriminator_steps': self.discriminator_steps,
            'gradient_penalty_weight': self.gradient_penalty_weight,
            'count_bars': self.count_bars,
            'count_tracks': self.count_tracks,
            'count_steps_per_bar': self.count_steps_per_bar,
            'count_notes': self.count_notes,
            'discriminator': self.discriminator,
            'generator': self.generator,
            'is_load': self.is_load
        }

    @property
    def metrics(self):
        return [self.discriminator_loss_metric,
                self.generator_loss_metric]

    def gradient_penalty(self, batch_size, real, fake):
        alpha = tf.random.normal([batch_size, 1, 1, 1, 1], 0.0, 1.0)
        difference = fake - real
        interpolated = real + alpha * difference

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
        return gradient_penalty

    def train_step(self, real):
        batch_size = tf.shape(real)[0]
        for i in range(self.discriminator_steps):
            chords_input = tf.random.normal(shape=(batch_size, self.noise_length))
            style_input = tf.random.normal(shape=(batch_size, self.noise_length))
            melody_input = tf.random.normal(shape=(batch_size, self.count_tracks, self.noise_length))
            groove_input = tf.random.normal(shape=(batch_size, self.count_tracks, self.noise_length))
            input = [chords_input, style_input, melody_input, groove_input]

            with tf.GradientTape() as tape:
                fake = self.generator(input, training=True)
                fake_preds = self.discriminator(fake, training=True)
                real_preds = self.discriminator(real, training=True)

                wasserstein_loss = tf.reduce_mean(fake_preds) - tf.reduce_mean(real_preds)
                gradient_penalty = self.gradient_penalty(batch_size, real, fake)
                discriminator_loss = wasserstein_loss + gradient_penalty * self.gradient_penalty_weight

            gradient = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradient, self.discriminator.trainable_variables))

        chords_input = tf.random.normal(shape=(batch_size, self.noise_length))
        style_input = tf.random.normal(shape=(batch_size, self.noise_length))
        melody_input = tf.random.normal(shape=(batch_size, self.count_tracks, self.noise_length))
        groove_input = tf.random.normal(shape=(batch_size, self.count_tracks, self.noise_length))
        input = [chords_input, style_input, melody_input, groove_input]

        with tf.GradientTape() as tape:
            fake = self.generator(input, training=True)
            fake_preds = self.discriminator(fake, training=True)
            generator_loss = -tf.reduce_mean(fake_preds)

        gradient = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradient, self.generator.trainable_variables))

        self.discriminator_loss_metric.update_state(discriminator_loss)
        self.generator_loss_metric.update_state(generator_loss)

        return {item.name: item.result() for item in self.metrics}

    def noise_reshaper_initialize(self):
        input = Input(shape=(self.noise_length,))
        reshape = Reshape([1, 1, self.noise_length])(input)

        conv_transpose1 = Conv2DTranspose(1024, (2, 1), (1, 1), "valid", kernel_initializer=self.initializer)(reshape)
        conv_transpose1 = BatchNormalization(momentum=0.9)(conv_transpose1)
        conv_transpose1 = Activation("relu")(conv_transpose1)
        conv_transpose2 = Conv2DTranspose(self.noise_length, (self.count_bars - 1, 1), (1, 1), "valid",
                                          kernel_initializer=self.initializer)(conv_transpose1)
        conv_transpose2 = BatchNormalization(momentum=0.9)(conv_transpose2)
        conv_transpose2 = Activation("relu")(conv_transpose2)

        output = Reshape([self.count_bars, self.noise_length])(conv_transpose2)
        return Model(input, output)

    def bar_generator_initialize(self):
        input = Input(shape=(self.noise_length * 4,))
        reshape = Dense(1024)(input)
        reshape = BatchNormalization(momentum=0.9)(reshape)
        reshape = Activation("relu")(reshape)
        reshape = Reshape([2, 1, 512])(reshape)

        conv_transpose1 = Conv2DTranspose(1024, (2, 1), (2, 1), "same", kernel_initializer=self.initializer)(reshape)
        conv_transpose1 = BatchNormalization(momentum=0.9)(conv_transpose1)
        conv_transpose1 = Activation("relu")(conv_transpose1)
        conv_transpose2 = Conv2DTranspose(512, (2, 1), (2, 1), "same", kernel_initializer=self.initializer)(conv_transpose1)
        conv_transpose2 = BatchNormalization(momentum=0.9)(conv_transpose2)
        conv_transpose2 = Activation("relu")(conv_transpose2)
        conv_transpose3 = Conv2DTranspose(256, (2, 1), (2, 1), "same", kernel_initializer=self.initializer)(conv_transpose2)
        conv_transpose3 = BatchNormalization(momentum=0.9)(conv_transpose3)
        conv_transpose3 = Activation("relu")(conv_transpose3)
        conv_transpose4 = Conv2DTranspose(256, (1, 5), (1, 5), "same", kernel_initializer=self.initializer)(conv_transpose3)
        conv_transpose4 = BatchNormalization(momentum=0.9)(conv_transpose4)
        conv_transpose4 = Activation("relu")(conv_transpose4)
        conv_transpose5 = Conv2DTranspose(256, (1, 5), (1, 5), "same", kernel_initializer=self.initializer)(conv_transpose4)
        conv_transpose5 = BatchNormalization(momentum=0.9)(conv_transpose5)
        conv_transpose5 = Activation("relu")(conv_transpose5)
        conv_transpose6 = Conv2DTranspose(1, (1, 3), (1, 3), "same", kernel_initializer=self.initializer)(conv_transpose5)
        conv_transpose6 = Activation("tanh")(conv_transpose6)

        output = Reshape([1, self.count_steps_per_bar, self.count_notes, 1])(conv_transpose6)
        return Model(input, output)

    def generator_initialize(self):
        chords = Input(shape=(self.noise_length,))
        style = Input(shape=(self.noise_length,))
        melody = Input(shape=(self.count_tracks, self.noise_length))
        groove = Input(shape=(self.count_tracks, self.noise_length))

        chords_reshaper = self.noise_reshaper_initialize()
        chords_reshaped = chords_reshaper(chords)

        melody_reshapers = []
        melody_reshaped = []
        for track_ix in range(self.count_tracks):
            melody_reshapers.append(self.noise_reshaper_initialize())
            melody_track = Lambda(lambda x, track_ix=track_ix: x[:, track_ix, :])(melody)
            melody_reshaped.append(melody_reshapers[track_ix](melody_track))

        bar_generators = []
        for track_ix in range(self.count_tracks):
            bar_generators.append(self.bar_generator_initialize())

        bars_output = []
        chord_noise = []
        for bar_ix in range(self.count_bars):
            chord_noise.append(Lambda(lambda x, bar_ix=bar_ix: x[:, bar_ix, :])(chords_reshaped))
            style_noise = style

            tracks_output = []
            for track_ix in range(self.count_tracks):
                melody_noise = Lambda(lambda x, bar_ix=bar_ix: x[:, bar_ix, :])(melody_reshaped[track_ix])
                groove_noise = Lambda(lambda x, track_ix=track_ix: x[:, track_ix, :])(groove)
                concat_input = Concatenate(axis=1)([chord_noise[bar_ix], style_noise, melody_noise, groove_noise])
                tracks_output.append(bar_generators[track_ix](concat_input))
            bars_output.append(Concatenate(axis=-1)(tracks_output))

        output = Concatenate(axis=1)(bars_output)
        return Model([chords, style, melody, groove], output)

    def discriminator_initialize(self):
        input = Input(shape=(self.count_bars, self.count_steps_per_bar, self.count_notes, self.count_tracks))

        conv1 = Conv3D(128, (2, 1, 1), (1, 1, 1), "valid")(input)
        conv1 = LeakyReLU()(conv1)
        conv2 = Conv3D(128, (self.count_bars - 1, 1, 1), (1, 1, 1), "valid")(conv1)
        conv2 = LeakyReLU()(conv2)
        conv3 = Conv3D(128, (1, 1, 3), (1, 1, 3), "same")(conv2)
        conv3 = LeakyReLU()(conv3)
        conv4 = Conv3D(128, (1, 1, 5), (1, 1, 5), "same")(conv3)
        conv4 = LeakyReLU()(conv4)
        conv5 = Conv3D(128, (1, 1, 5), (1, 1, 5), "same")(conv4)
        conv5 = LeakyReLU()(conv5)
        conv6 = Conv3D(128, (1, 2, 1), (1, 2, 1), "same")(conv5)
        conv6 = LeakyReLU()(conv6)
        conv7 = Conv3D(128, (1, 2, 1), (1, 2, 1), "same")(conv6)
        conv7 = LeakyReLU()(conv7)
        conv8 = Conv3D(256, (1, 2, 1), (1, 2, 1), "same")(conv7)
        conv8 = LeakyReLU()(conv8)
        conv9 = Conv3D(512, (1, 2, 1), (1, 2, 1), "same")(conv8)
        conv9 = LeakyReLU()(conv9)

        flatten = Flatten()(conv9)
        dense = Dense(1024, kernel_initializer=self.initializer)(flatten)
        dense = LeakyReLU()(dense)

        output = Dense(1, None, kernel_initializer=self.initializer)(dense)
        return Model(input, output)

class MuseGAN(Model):
    def __init__(self,
                 noise_length=32,
                 discriminator_steps=5,
                 gradient_penalty_weight=10,
                 count_bars=9,
                 count_tracks=4,
                 count_steps_per_bar=16,
                 count_notes=75,
                 is_load=False,
                 trainable=True,
                 dtype='float32'):
        super(MuseGAN, self).__init__()
        self.initializer = RandomNormal(mean=0.0, stddev=0.02)
        self.noise_length = noise_length
        self.discriminator_steps = discriminator_steps
        self.gradient_penalty_weight = gradient_penalty_weight
        self.count_bars = count_bars
        self.count_tracks = count_tracks
        self.count_steps_per_bar = count_steps_per_bar
        self.count_notes = count_notes
        self.discriminator = self.discriminator_initialize()
        self.generator = self.generator_initialize()
        self.is_load = is_load

    def compile(self, discriminator_learning_rate, generator_learning_rate, adam_beta1, adam_beta2):
        #if (self.is_load):
        #    self.load_weights("./weights/checkpoint.weights.h5")

        super(MuseGAN, self).compile()
        self.discriminator_optimizer = Adam(discriminator_learning_rate, adam_beta1, adam_beta2)
        self.generator_optimizer = Adam(generator_learning_rate, adam_beta1, adam_beta2)
        self.discriminator_loss_metric = Mean(name="discriminator_loss")
        self.generator_loss_metric = Mean(name="generator_loss")

    def get_config(self):
        return {
            'initializer': self.initializer,
            'noise_length': self.noise_length,
            'discriminator_steps': self.discriminator_steps,
            'gradient_penalty_weight': self.gradient_penalty_weight,
            'count_bars': self.count_bars,
            'count_tracks': self.count_tracks,
            'count_steps_per_bar': self.count_steps_per_bar,
            'count_notes': self.count_notes,
            'discriminator': self.discriminator,
            'generator': self.generator,
            'is_load': self.is_load
        }

    @property
    def metrics(self):
        return [self.discriminator_loss_metric,
                self.generator_loss_metric]

    def gradient_penalty(self, batch_size, real, fake):
        alpha = tf.random.normal([batch_size, 1, 1, 1, 1], 0.0, 1.0)
        difference = fake - real
        interpolated = real + alpha * difference

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
        return gradient_penalty

    def train_step(self, real):
        batch_size = tf.shape(real)[0]
        for i in range(self.discriminator_steps):
            chords_input = tf.random.normal(shape=(batch_size, self.noise_length))
            style_input = tf.random.normal(shape=(batch_size, self.noise_length))
            melody_input = tf.random.normal(shape=(batch_size, self.count_tracks, self.noise_length))
            groove_input = tf.random.normal(shape=(batch_size, self.count_tracks, self.noise_length))
            input = [chords_input, style_input, melody_input, groove_input]

            with tf.GradientTape() as tape:
                fake = self.generator(input, training=True)
                fake_preds = self.discriminator(fake, training=True)
                real_preds = self.discriminator(real, training=True)

                wasserstein_loss = tf.reduce_mean(fake_preds) - tf.reduce_mean(real_preds)
                gradient_penalty = self.gradient_penalty(batch_size, real, fake)
                discriminator_loss = wasserstein_loss + gradient_penalty * self.gradient_penalty_weight

            gradient = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradient, self.discriminator.trainable_variables))

        chords_input = tf.random.normal(shape=(batch_size, self.noise_length))
        style_input = tf.random.normal(shape=(batch_size, self.noise_length))
        melody_input = tf.random.normal(shape=(batch_size, self.count_tracks, self.noise_length))
        groove_input = tf.random.normal(shape=(batch_size, self.count_tracks, self.noise_length))
        input = [chords_input, style_input, melody_input, groove_input]

        with tf.GradientTape() as tape:
            fake = self.generator(input, training=True)
            fake_preds = self.discriminator(fake, training=True)
            generator_loss = -tf.reduce_mean(fake_preds)

        gradient = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradient, self.generator.trainable_variables))

        self.discriminator_loss_metric.update_state(discriminator_loss)
        self.generator_loss_metric.update_state(generator_loss)

        return {item.name: item.result() for item in self.metrics}

    def noise_reshaper_initialize(self):
        input = Input(shape=(self.noise_length,))
        reshape = Reshape([1, 1, self.noise_length])(input)

        conv_transpose1 = Conv2DTranspose(1024, (2, 1), (1, 1), "valid", kernel_initializer=self.initializer)(reshape)
        conv_transpose1 = BatchNormalization(momentum=0.9)(conv_transpose1)
        conv_transpose1 = Activation("relu")(conv_transpose1)
        conv_transpose2 = Conv2DTranspose(self.noise_length, (self.count_bars - 1, 1), (1, 1), "valid",
                                          kernel_initializer=self.initializer)(conv_transpose1)
        conv_transpose2 = BatchNormalization(momentum=0.9)(conv_transpose2)
        conv_transpose2 = Activation("relu")(conv_transpose2)

        output = Reshape([self.count_bars, self.noise_length])(conv_transpose2)
        return Model(input, output)

    def bar_generator_initialize(self):
        input = Input(shape=(self.noise_length * 4,))
        reshape = Dense(1024)(input)
        reshape = BatchNormalization(momentum=0.9)(reshape)
        reshape = Activation("relu")(reshape)
        reshape = Reshape([2, 1, 512])(reshape)

        conv_transpose1 = Conv2DTranspose(1024, (2, 1), (2, 1), "same", kernel_initializer=self.initializer)(reshape)
        conv_transpose1 = BatchNormalization(momentum=0.9)(conv_transpose1)
        conv_transpose1 = Activation("relu")(conv_transpose1)
        conv_transpose2 = Conv2DTranspose(512, (2, 1), (2, 1), "same", kernel_initializer=self.initializer)(conv_transpose1)
        conv_transpose2 = BatchNormalization(momentum=0.9)(conv_transpose2)
        conv_transpose2 = Activation("relu")(conv_transpose2)
        conv_transpose3 = Conv2DTranspose(256, (2, 1), (2, 1), "same", kernel_initializer=self.initializer)(conv_transpose2)
        conv_transpose3 = BatchNormalization(momentum=0.9)(conv_transpose3)
        conv_transpose3 = Activation("relu")(conv_transpose3)
        conv_transpose4 = Conv2DTranspose(256, (1, 5), (1, 5), "same", kernel_initializer=self.initializer)(conv_transpose3)
        conv_transpose4 = BatchNormalization(momentum=0.9)(conv_transpose4)
        conv_transpose4 = Activation("relu")(conv_transpose4)
        conv_transpose5 = Conv2DTranspose(256, (1, 5), (1, 5), "same", kernel_initializer=self.initializer)(conv_transpose4)
        conv_transpose5 = BatchNormalization(momentum=0.9)(conv_transpose5)
        conv_transpose5 = Activation("relu")(conv_transpose5)
        conv_transpose6 = Conv2DTranspose(1, (1, 3), (1, 3), "same", kernel_initializer=self.initializer)(conv_transpose5)
        conv_transpose6 = Activation("tanh")(conv_transpose6)

        output = Reshape([1, self.count_steps_per_bar, self.count_notes, 1])(conv_transpose6)
        return Model(input, output)

    def generator_initialize(self):
        chords = Input(shape=(self.noise_length,))
        style = Input(shape=(self.noise_length,))
        melody = Input(shape=(self.count_tracks, self.noise_length))
        groove = Input(shape=(self.count_tracks, self.noise_length))

        chords_reshaper = self.noise_reshaper_initialize()
        chords_reshaped = chords_reshaper(chords)

        melody_reshapers = []
        melody_reshaped = []
        for track_ix in range(self.count_tracks):
            melody_reshapers.append(self.noise_reshaper_initialize())
            melody_track = Lambda(lambda x, track_ix=track_ix: x[:, track_ix, :])(melody)
            melody_reshaped.append(melody_reshapers[track_ix](melody_track))

        bar_generators = []
        for track_ix in range(self.count_tracks):
            bar_generators.append(self.bar_generator_initialize())

        bars_output = []
        chord_noise = []
        for bar_ix in range(self.count_bars):
            chord_noise.append(Lambda(lambda x, bar_ix=bar_ix: x[:, bar_ix, :])(chords_reshaped))
            style_noise = style

            tracks_output = []
            for track_ix in range(self.count_tracks):
                melody_noise = Lambda(lambda x, bar_ix=bar_ix: x[:, bar_ix, :])(melody_reshaped[track_ix])
                groove_noise = Lambda(lambda x, track_ix=track_ix: x[:, track_ix, :])(groove)
                concat_input = Concatenate(axis=1)([chord_noise[bar_ix], style_noise, melody_noise, groove_noise])
                tracks_output.append(bar_generators[track_ix](concat_input))
            bars_output.append(Concatenate(axis=-1)(tracks_output))

        output = Concatenate(axis=1)(bars_output)
        return Model([chords, style, melody, groove], output)

    def discriminator_initialize(self):
        input = Input(shape=(self.count_bars, self.count_steps_per_bar, self.count_notes, self.count_tracks))

        conv1 = Conv3D(128, (2, 1, 1), (1, 1, 1), "valid")(input)
        conv1 = LeakyReLU()(conv1)
        conv2 = Conv3D(128, (self.count_bars - 1, 1, 1), (1, 1, 1), "valid")(conv1)
        conv2 = LeakyReLU()(conv2)
        conv3 = Conv3D(128, (1, 1, 3), (1, 1, 3), "same")(conv2)
        conv3 = LeakyReLU()(conv3)
        conv4 = Conv3D(128, (1, 1, 5), (1, 1, 5), "same")(conv3)
        conv4 = LeakyReLU()(conv4)
        conv5 = Conv3D(128, (1, 1, 5), (1, 1, 5), "same")(conv4)
        conv5 = LeakyReLU()(conv5)
        conv6 = Conv3D(128, (1, 2, 1), (1, 2, 1), "same")(conv5)
        conv6 = LeakyReLU()(conv6)
        conv7 = Conv3D(128, (1, 2, 1), (1, 2, 1), "same")(conv6)
        conv7 = LeakyReLU()(conv7)
        conv8 = Conv3D(256, (1, 2, 1), (1, 2, 1), "same")(conv7)
        conv8 = LeakyReLU()(conv8)
        conv9 = Conv3D(512, (1, 2, 1), (1, 2, 1), "same")(conv8)
        conv9 = LeakyReLU()(conv9)

        flatten = Flatten()(conv9)
        dense = Dense(1024, kernel_initializer=self.initializer)(flatten)
        dense = LeakyReLU()(dense)

        output = Dense(1, None, kernel_initializer=self.initializer)(dense)
        return Model(input, output)