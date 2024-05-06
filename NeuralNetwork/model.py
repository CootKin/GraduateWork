import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2DTranspose, Conv3D, Flatten, BatchNormalization, Activation, LeakyReLU, Reshape, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean


class MuseGAN(Model):
    def __init__(self, discriminator, generator, noise_length, count_tracks, discriminator_steps, gradient_penalty_weight):
        super(MuseGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_length = noise_length
        self.count_tracks = count_tracks
        self.discriminator_steps = discriminator_steps
        self.gradient_penalty_weight = gradient_penalty_weight

    def compile(self, discriminator_optimizer, generator_optimizer):
        super(MuseGAN, self).compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator_loss_metric = Mean(name="discriminator_loss")
        self.generator_loss_metric = Mean(name="generator_loss")

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


def noise_reshaper_initialize(initializer, noise_length, count_bars):
    input = Input(shape=(noise_length,))
    reshape = Reshape([1, 1, noise_length])(input)

    conv_transpose1 = Conv2DTranspose(1024, (2, 1), (1, 1), "valid", kernel_initializer=initializer)(reshape)
    conv_transpose1 = BatchNormalization(momentum=0.9)(conv_transpose1)
    conv_transpose1 = Activation("relu")(conv_transpose1)
    conv_transpose2 = Conv2DTranspose(noise_length, (count_bars - 1, 1), (1, 1), "valid",
                                      kernel_initializer=initializer)(conv_transpose1)
    conv_transpose2 = BatchNormalization(momentum=0.9)(conv_transpose2)
    conv_transpose2 = Activation("relu")(conv_transpose2)

    output = Reshape([count_bars, noise_length])(conv_transpose2)
    return Model(input, output)


def bar_generator_initialize(initializer, noise_length, count_steps_per_bar, count_notes):
    input = Input(shape=(noise_length * 4,))
    reshape = Dense(1024)(input)
    reshape = BatchNormalization(momentum=0.9)(reshape)
    reshape = Activation("relu")(reshape)
    reshape = Reshape([2, 1, 512])(reshape)

    conv_transpose1 = Conv2DTranspose(1024, (2, 1), (2, 1), "same", kernel_initializer=initializer)(reshape)
    conv_transpose1 = BatchNormalization(momentum=0.9)(conv_transpose1)
    conv_transpose1 = Activation("relu")(conv_transpose1)
    conv_transpose2 = Conv2DTranspose(512, (2, 1), (2, 1), "same", kernel_initializer=initializer)(conv_transpose1)
    conv_transpose2 = BatchNormalization(momentum=0.9)(conv_transpose2)
    conv_transpose2 = Activation("relu")(conv_transpose2)
    conv_transpose3 = Conv2DTranspose(256, (2, 1), (2, 1), "same", kernel_initializer=initializer)(conv_transpose2)
    conv_transpose3 = BatchNormalization(momentum=0.9)(conv_transpose3)
    conv_transpose3 = Activation("relu")(conv_transpose3)
    conv_transpose4 = Conv2DTranspose(256, (1, 5), (1, 5), "same", kernel_initializer=initializer)(conv_transpose3)
    conv_transpose4 = BatchNormalization(momentum=0.9)(conv_transpose4)
    conv_transpose4 = Activation("relu")(conv_transpose4)
    conv_transpose5 = Conv2DTranspose(256, (1, 5), (1, 5), "same", kernel_initializer=initializer)(conv_transpose4)
    conv_transpose5 = BatchNormalization(momentum=0.9)(conv_transpose5)
    conv_transpose5 = Activation("relu")(conv_transpose5)
    conv_transpose6 = Conv2DTranspose(1, (1, 3), (1, 3), "same", kernel_initializer=initializer)(conv_transpose5)
    conv_transpose6 = Activation("tanh")(conv_transpose6)

    output = Reshape([1, count_steps_per_bar, count_notes, 1])(conv_transpose6)
    return Model(input, output)


def generator_initialize(initializer, noise_length, count_tracks, count_bars, count_steps_per_bar, count_notes):
    chords = Input(shape=(noise_length,))
    style = Input(shape=(noise_length,))
    melody = Input(shape=(count_tracks, noise_length))
    groove = Input(shape=(count_tracks, noise_length))

    chords_reshaper = noise_reshaper_initialize(
        initializer, noise_length, count_bars
    )
    chords_reshaped = chords_reshaper(chords)

    melody_reshapers = []
    melody_reshaped = []
    for track_ix in range(count_tracks):
        melody_reshapers.append(noise_reshaper_initialize(
            initializer, noise_length, count_bars
        ))
        melody_track = Lambda(lambda x, track_ix=track_ix: x[:, track_ix, :])(melody)
        melody_reshaped.append(melody_reshapers[track_ix](melody_track))

    bar_generators = []
    for track_ix in range(count_tracks):
        bar_generators.append(bar_generator_initialize(
            initializer, noise_length, count_steps_per_bar, count_notes
        ))

    bars_output = []
    chord_noise = []
    for bar_ix in range(count_bars):
        chord_noise.append(Lambda(lambda x, bar_ix=bar_ix: x[:, bar_ix, :])(chords_reshaped))
        style_noise = style

        tracks_output = []
        for track_ix in range(count_tracks):
            melody_noise = Lambda(lambda x, bar_ix=bar_ix: x[:, bar_ix, :])(melody_reshaped[track_ix])
            groove_noise = Lambda(lambda x, track_ix=track_ix: x[:, track_ix, :])(groove)
            concat_input = Concatenate(axis=1)([chord_noise[bar_ix], style_noise, melody_noise, groove_noise])
            tracks_output.append(bar_generators[track_ix](concat_input))
        bars_output.append(Concatenate(axis=-1)(tracks_output))

    output = Concatenate(axis=1)(bars_output)
    return Model([chords, style, melody, groove], output)


def discriminator_initialize(initializer, count_bars, count_steps_per_bar, count_notes, count_tracks):
    input = Input(shape=(count_bars, count_steps_per_bar, count_notes, count_tracks))

    conv1 = Conv3D(128, (2, 1, 1), (1, 1, 1), "valid")(input)
    conv1 = LeakyReLU()(conv1)
    conv2 = Conv3D(128, (count_bars - 1, 1, 1), (1, 1, 1), "valid")(conv1)
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
    dense = Dense(1024, kernel_initializer=initializer)(flatten)
    dense = LeakyReLU()(dense)

    output = Dense(1, None, kernel_initializer=initializer)(dense)
    return Model(input, output)