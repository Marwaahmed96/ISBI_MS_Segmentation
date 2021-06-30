import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Dropout, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Conv3DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

class U_Net_Model(tf.keras.Model):
    '''
    conv block
        - convolution layer
        - activation
        - batch normalization layer, instance normalization layer
        - drop out
        - convolution layer
        - activation
        - max pooling layer, average pooling layer

    '''
    def __init__(self, model_input_shape,
                 out_channel=1, 
                 pooling_kernel=(2, 2, 2),
                 drop_out=False,
                 drop_rate=0.1,
                 initial_learning_rate=0.00001,
                 deconvolution=False, 
                 depth=4, 
                 n_base_filters=32,
                 kernel=(3,3,3),
                 padding='same',
                 strides=(1,1,1),
                 up_strides=(2,2,2),
                 up_kernel=(2,2,2),
                 max_pooling=False,
                 batch_normalization=False,
                 instance_normalization=False,
                 activation="sigmoid"):
        super().__init__()
        self.model_input_shape = model_input_shape
        self.out_channel=out_channel
        self.pooling_kernel = pooling_kernel
        self.drop_out=drop_out
        self.drop_rate=drop_rate
        self.initial_learning_rate = initial_learning_rate
        self.deconvolution = deconvolution
        self.depth = depth
        self.n_base_filters = n_base_filters
        self.kernel=kernel
        self.padding=padding
        self.strides=strides
        self.up_strides=up_strides
        self.up_kernel=up_kernel
        self.max_pooling=max_pooling
        self.batch_normalization = batch_normalization
        self.instance_normalization=instance_normalization
        self.activation = activation

        self.unet_model = self.unet_model_3d(
            input_shape=self.model_input_shape ,
            out_channel=self.out_channel,
            pooling_kernel=self.pooling_kernel ,
            drop_out=self.drop_out,
            drop_rate=self.drop_rate,
            initial_learning_rate=self.initial_learning_rate ,
            deconvolution=self.deconvolution ,
            depth=self.depth ,
            n_base_filters=self.n_base_filters ,
            kernel=self.kernel,
            padding=self.padding,
            strides=self.strides,
            up_strides=self.up_strides,
            up_kernel=self.up_kernel,
            max_pooling=self.max_pooling,
            batch_normalization=self.batch_normalization ,
            instance_normalization=self.instance_normalization,
            activation=self.activation )
        
    def call(self, inputs):
        return self.unet_model(inputs)

    
    def train_step22(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        #print("x",x)
        #print("x",K.int_shape(x),"y",K.int_shape(y))
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def create_convolution_block(self,input_layer, 
                                 n_filters, 
                                 batch_normalization,
                                 drop_out,
                                 drop_rate,
                                 kernel, 
                                 padding, 
                                 strides,
                                 max_pooling, 
                                 pooling_kernel, 
                                 instance_normalization,
                                activation=None):
        """
        :param input_layer:
        :param n_filters:
        :param batch_normalization:
        :param kernel:
        :param activation: Keras activation layer to use. (default is 'relu')
        :param padding:
        :return:
        """
        # conv layer
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)

        # activation layer
        if activation is None:
            layer= Activation('relu')(layer)
        else:
            layer= activation()(layer)

        # batch normalization or instance normalization layer
        if batch_normalization:
            layer = BatchNormalization(axis=1)(layer)
        elif instance_normalization:
            try:
                from keras_contrib.layers.normalization import InstanceNormalization
            except ImportError:
                raise ImportError("Install keras_contrib in order to use instance normalization."
                                  "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
            layer = InstanceNormalization(axis=1)(layer)

        # drop out layer
        if drop_out:
            layer=Dropout(drop_rate)(layer)

        # conv layer
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(layer)

        # activation layer
        if activation is None:
            layer= Activation('relu')(layer)
        else:
            layer= activation()(layer)

        '''
        
        # max pooling layer
        if max_pooling:
            layer= MaxPooling3D(pooling_kernel)(layer)
        '''
        return layer
        
    
    def get_up_convolution(self, n_filters, 
                           pooling_kernel, 
                           kernel, 
                           strides, 
                           deconvolution):
        if deconvolution:
            return Conv3DTranspose(filters=n_filters, kernel_size=kernel, strides=strides)
        else:
            return UpSampling3D(size=pooling_kernel)

    # metrics=dice_coefficient
    def unet_model_3d(self, input_shape,
                      out_channel=3, 
                      pooling_kernel=(2, 2, 2),
                      drop_out=False,
                      drop_rate=0.1,
                      initial_learning_rate=0.00001,
                      deconvolution=False, 
                      depth=4, 
                      n_base_filters=32,
                      kernel=(3,3,3),
                      padding='same',
                      strides=(1, 1, 1),
                      up_strides=(2,2,2),
                      up_kernel=(2,2,2),
                      max_pooling=False,
                      batch_normalization=False, 
                      instance_normalization=False,
                      activation="sigmoid"):
        
        """
        Builds the 3D UNet Keras model.f
        :param metrics: List metrics to be calculated during model training (default is dice coefficient).
        :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
        layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
        to train the model.
        :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
        layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
        :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
        divisible by the pool size to the power of the depth of the UNet, that is pooling_kernel^depth.
        :param pooling_kernel: Pool size for the max pooling operations.
        :param n_labels: Number of binary labels that the model is learning.
        :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
        :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
        increases the amount memory required during training.
        :return: Untrained 3D UNet Model
        """
        inputs = Input(input_shape)
        current_layer = inputs
        levels = []
        print(f'input shape {inputs.shape}')
        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = self.create_convolution_block(input_layer=current_layer, 
                                                   n_filters=n_base_filters*(2**layer_depth),
                                                   batch_normalization=batch_normalization,
                                                   drop_out=drop_out,
                                                   drop_rate=drop_rate,
                                                   kernel=kernel,
                                                   
                                                   padding=padding,
                                                   strides=strides,
                                                   max_pooling=max_pooling,
                                                   pooling_kernel=pooling_kernel,
                                                   instance_normalization=instance_normalization,
                                                  activation=None)
            
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=pooling_kernel)(layer1)
                levels.append([layer1, current_layer])
            else:
                current_layer = layer1
                levels.append([layer1])
        # add levels with up-convolution or up-sampling
        for layer_depth in range(depth-2, -1, -1):
            
            up_convolution = self.get_up_convolution(n_filters=current_layer.shape[1],
                                                     pooling_kernel=pooling_kernel, 
                                                     kernel=up_kernel,
                                                     strides=up_strides,
                                                     deconvolution=deconvolution)(current_layer)
            concat = concatenate([up_convolution, levels[layer_depth][0]], axis=4)
            current_layer = self.create_convolution_block(input_layer=concat, 
                                                   n_filters=levels[layer_depth][0].shape[-1],
                                                   batch_normalization=batch_normalization,
                                                   drop_out=drop_out,
                                                   drop_rate=drop_rate,
                                                   kernel=kernel,
                                                   
                                                   padding=padding,
                                                   strides=strides,
                                                   max_pooling=max_pooling,
                                                   pooling_kernel=pooling_kernel,
                                                   instance_normalization=instance_normalization,
                                                         activation=None)

        # number of labels: 1
        final_convolution = Conv3D(out_channel, (1, 1, 1))(current_layer)
        act = Activation(activation)(final_convolution)
        model = Model(inputs=inputs, outputs=act)

        #model.summary()
        return model
    
    def get_callbacks(self,weights_file_path, fold, initial_learning_rate=0.0001, learning_rate_drop=0.5,
                      learning_rate_patience=50, verbosity=1, early_stopping_patience=None):

        check_point = ModelCheckpoint(weights_file_path+'fold_' + fold + '_weights-{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
        csv_log = CSVLogger(weights_file_path+'training-log.csv', append=True)

        # potential problem of recude learning rate: https://github.com/keras-team/keras/issues/10924
        reduce = ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience, verbose=verbosity)
        if early_stopping_patience:
            early_stop = EarlyStopping(verbose=verbosity, patience=early_stopping_patience)
            return [check_point, csv_log, reduce, early_stop]
        else:
            return [check_point, csv_log, reduce]