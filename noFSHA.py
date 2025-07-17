import tensorflow as tf
import numpy as np
import tqdm
import logging
import datasets, architectures
import os
import time
import datetime


class FSHA:
    
    def loadBiasNetwork(self, make_decoder, z_shape, channels):
        return make_decoder(z_shape, channels=channels)
        
    def __init__(self, xpriv, xpub, id_setup, batch_size, hparams):
        logdir = 'logs_training'
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        while True:
            log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
            log_path = log_file_name+'.log'
        
            if not os.path.exists(os.path.join(logdir, log_path)):
                break
            else:
                time.sleep(1)
        
        logging.basicConfig(
            filename=os.path.join(logdir, log_path),
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w'
        )

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.info("#" * 100)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


        input_shape = xpriv.element_spec[0].shape
        
        self.hparams = hparams

        # setup dataset
        self.client_dataset = xpriv.batch(batch_size, drop_remainder=True).repeat(-1)
        self.batch_size = batch_size

        ## setup models
        make_f, make_tilde_f, make_decoder, make_D = architectures.SETUPS[id_setup]

        self.f = make_f(input_shape)
        self.tilde_f = make_tilde_f(input_shape)

        assert self.f.output.shape.as_list()[1:] == self.tilde_f.output.shape.as_list()[1:]
        z_shape = self.tilde_f.output.shape.as_list()[1:]

        self.D = make_D(z_shape)

        # setup optimizers
        self.optimizer0 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_f'])
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_D'])

    @tf.function
    def train_step(self, x_private, label_private):
        with tf.GradientTape(persistent=True) as tape:
            #### Virtually, ON THE CLIENT SIDE:
            # clients' smashed data
            z_private = self.f(x_private, training=True)
            ####################################
            tf.debugging.check_numerics(z_private, message="NaN or Inf in z")


            #### SERVER-SIDE:
            # map to data space (for evaluation and style loss)
            adv_private_logits = self.D(z_private, training=True)
            tf.debugging.check_numerics(adv_private_logits, message="NaN or Inf in y_pred")
            prediction = tf.cast(tf.argmax(adv_private_logits, axis=1), tf.int32)
            # tf.print(label_private)
            label_private_onehot = tf.one_hot(label_private, depth=10, axis=1)
            # tf.print(label_private_onehot)
            label_private_casted = tf.cast(label_private, tf.int32)
            correct_prediction = tf.equal(prediction, label_private_casted)

            D_loss = self.loss_fn(label_private_onehot, adv_private_logits) 

        # Compute gradients
        grads_server = tape.gradient(D_loss, self.D.trainable_variables)
        dz = tape.gradient(D_loss, z_private)  # Gradient of loss w.r.t. client output

        # Backpropagate into client
        grads_client = tape.gradient(z_private, self.f.trainable_variables, output_gradients=dz)

        # Apply gradients
        self.optimizer0.apply_gradients(zip(grads_server, self.D.trainable_variables))
        self.optimizer2.apply_gradients(zip(grads_client, self.f.trainable_variables))


        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return  D_loss, accuracy

    # EDWWWW
    def __call__(self, iterations, log_frequency=500, verbose=False, progress_bar=True):

        n = int(iterations / log_frequency)
        LOG = np.zeros((n, 2))

        iterator = self.client_dataset.take(iterations)
        if progress_bar:
            iterator = tqdm.tqdm(iterator , total=iterations)

        i, j = 0, 0
        self.logger.info("RUNNING...")
        for x_private, label_private in iterator:
            log = self.train_step(x_private, label_private)
            if i == 0:
                VAL = log[0] 
                VAL_A = log[1]
            else:
                VAL += log[0] / log_frequency
                VAL_A += log[1] / log_frequency

            if  i % log_frequency == 0:
                LOG[j] = log

                if verbose:
                    self.logger.info("log--%02d%%-%07d] validation: %0.4f acc: %0.4f" % ( int(i/iterations*100) ,i, VAL, VAL_A) )

                VAL = 0
                VAL_A = 0
                j += 1


            i += 1
        return LOG

