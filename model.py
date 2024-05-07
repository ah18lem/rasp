import tensorflow as tf
from keras.layers import Lambda
import functions
import constantes

class AutoencoderWithClassifier(tf.keras.Model):
    def __init__(
        self,
        input_dim=None,
        isServer=True,
        encoder_layer_sizes=constantes.ENCODER_LAYERS,
        decoder_layer_sizes=constantes.DECODER_LAYERS,
        seed=42,
        vae=False,
    ):
        super(AutoencoderWithClassifier, self).__init__()

        # Set a random seed for reproducibility
        initialiser = tf.keras.initializers.GlorotUniform(seed=seed)

        input_layer = tf.keras.layers.Input(shape=(input_dim,), batch_size=None)
        x = input_layer
       

        if vae:
            # Create VAE encoder
            x, latent_mean, latent_log_var = self.build_vae_encoder(
                x, encoder_layer_sizes, initialiser, isServer
            )
            self.latent_mean = latent_mean
            self.latent_log_var = latent_log_var
            self.z = x

        else:
            # Create encoder
            x = self.build_encoder(x, encoder_layer_sizes, initialiser, isServer)
            self.latent = x

        encoded = x

        # Create decoder layers
        x = encoded
        for i, layer_size in enumerate(decoder_layer_sizes):
            x = tf.keras.layers.Dense(
                layer_size,
                activation="relu",
                trainable=(not isServer),  # Freeze layers if isServer is True
                name=f"decoder{i + 1}",
                kernel_initializer=initialiser,
            )(x)

        x = tf.keras.layers.Dense(
            input_dim,
            activation="relu",
            name="decoder",
            trainable=(not isServer),  # Freeze layers if isServer is True
            kernel_initializer=initialiser,
        )(x)
        decoded = x

        # Define the classification layer
        activation = "softmax"  # Always use softmax activation
        classification_layer = tf.keras.layers.Dense(
            constantes.NUM_CLASSES,  # Assuming constantes.NUM_CLASSES represents the number of classes
            activation=activation,
            name="classification_layer",
            kernel_initializer=initialiser,
            trainable=isServer,  # Freeze layer if isServer is True
        )(encoded)

        # Define losses and compile the model
        loss = "categorical_crossentropy"  # Always use categorical_crossentropy for multi-class classification
        loss_weights = [1.0, 0.0] if isServer else [0.0, 1.0]
        
        # Define the classification model
        classification_model = tf.keras.models.Model(
            inputs=input_layer, outputs=[classification_layer, decoded]
        )
        if isServer:
            # Server: Only train the classification layer
            classification_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=constantes.LEARNING_RATE),
                loss=[loss, "mean_squared_error"],
                loss_weights=loss_weights,
            )
        else:
            # Client: Train based on architecture (AE or VAE)
            if vae:
                print("CLIENT VAE")
                # Add KL divergence loss to the model
                def kl_loss(z_mean, z_log_var):
                    kl_loss = -5e-4 * tf.keras.backend.mean(
                        1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1
                    )
                    return kl_loss
                kl = kl_loss(self.latent_mean, self.latent_log_var)     
                classification_model.add_loss(kl)
                classification_model.add_metric(kl, name="kl_loss", aggregation="mean")

                # Add reconstruction loss for VAE
                reconstruction_loss = tf.keras.losses.mean_squared_error(
                    input_layer, decoded
                )
                classification_model.add_loss(reconstruction_loss)
                classification_model.add_metric(
                    reconstruction_loss, name="mse_loss", aggregation="mean"
                )
                classification_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=constantes.LEARNING_RATE),
                )

            else:
                print("CLIENT AE")
                # Autoencoder: Apply mean squared error loss
                classification_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=constantes.LEARNING_RATE),
                    loss=[loss, "mean_squared_error"],
                    loss_weights=loss_weights,
                )

        self.model = classification_model
        self.vae = vae
        self.isServer = isServer

    def call(self, inputs):
        classification, decoded = self.model(inputs)
        return classification, decoded

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def train(self, train_data, train_labels, epochs):
        # Freeze the autoencoder layers
       
        self.model.fit(
            train_data,
            [train_labels, train_data],
            epochs=epochs,
            batch_size=constantes.BATCH_SIZE,
            shuffle=False,
        )

    def build_encoder(self, x, encoder_layer_sizes, initialiser, isServer):
        # Create encoder layers
        for i, layer_size in enumerate(encoder_layer_sizes):
            x = tf.keras.layers.Dense(
                layer_size,
                activation="relu",
                trainable=(not isServer),
                name=f"encoder{i + 1}",
                kernel_initializer=initialiser,
            )(x)
        return x

    def build_vae_encoder(self, x, encoder_layer_sizes, initialiser, isServer):
        # Create VAE encoder layers
        for i, layer_size in enumerate(encoder_layer_sizes[:-1]):  # Skip the last layer
            x = tf.keras.layers.Dense(
                layer_size,
                activation="relu",
                trainable=(not isServer),
                name=f"encoder{i + 1}",
                kernel_initializer=initialiser,
            )(x)

        # Create VAE latent mean and variance layers
        latent_mean = tf.keras.layers.Dense(
            20,
            activation=None,
            name="latent_mean",
            trainable=(not isServer),
            kernel_initializer=initialiser,
        )(x)

        latent_log_var = tf.keras.layers.Dense(
            20,
            activation=None,
            name="latent_log_var",
            trainable=(not isServer),
            kernel_initializer=initialiser,
            
        )(x)

        # Reparameterization trick for VAE

        z = Lambda(lambda args: self.sample_z(*args), output_shape=(20,), name="z")(
            [latent_mean, latent_log_var]
        )

        return z, latent_mean, latent_log_var

    def sample_z(self, latent_mean, latent_log_var):
        eps = tf.keras.backend.random_normal(
            shape=(tf.shape(latent_mean)[0], tf.keras.backend.int_shape(latent_mean)[1])
        )
        return latent_mean + tf.exp(latent_log_var / 2) * eps
