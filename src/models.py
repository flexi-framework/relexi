#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.specs import tensor_spec
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils


def tanh_squash_to_spec(inputs, spec):
  """Maps inputs with arbitrary range to range defined by spec using `tanh`."""
  means = (spec.maximum + spec.minimum) / 2.0
  magnitudes = (spec.maximum - spec.minimum) / 2.0
  return means + magnitudes * tf.tanh(inputs)


class ActionNetCNN(network.Network):
  """
  Policy network architecture purely built from 3D convolutional layers
  ATTENTION: Might be incomparible for certain choices for the kernel size and
             the polyinomial degree N of FLEXI!
  """
  def __init__(self
              ,input_tensor_spec
              ,output_tensor_spec
              ,kernel=3
              ,action_std=0.02
              ,debug=0):
    super(ActionNetCNN, self).__init__(
        input_tensor_spec = input_tensor_spec,
        state_spec=(),
        name='ActionNetCNN')
    self._output_tensor_spec = output_tensor_spec
    self._action_std = action_std

    # Build model
    self.kernel=kernel
    self._buildmodel(kernel,debug)

  def call(self, observations, step_type, network_state):
    del step_type

    # We use batch_squash here in case the observations have a time sequence compoment.
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    batch_squash = utils.BatchSquash(outer_rank)
    observations = tf.nest.map_structure(batch_squash.flatten, observations)

    # Evaluate model
    action_means = self._model(observations)

    # Unsquash in time
    action_means = batch_squash.unflatten(action_means)

    # Scale to specs
    #action_means = tanh_squash_to_spec(action_means, self._output_tensor_spec)

    # Get standard deviation for actions
    action_std = tf.ones_like(action_means)*self._action_std

    # Return distribution
    return tfp.distributions.MultivariateNormalDiag(action_means, action_std), network_state

  def _buildmodel(self,kernel,debug):
    input_shape = self._input_tensor_spec.shape
    Np1 = input_shape[-2]

    # Determine network architecture, which results in scalar output for given kernelsize and N
    n_layers    = int(np.floor((Np1-1)/(kernel-1))) # Times to apply standard kernelsize
    last_kernel = Np1 - n_layers * (kernel-1)  # "Remainder": kernelsize for last layer
    # If last_kernel is zero, set it to standard kernelsize and reduce n_layers by one
    if last_kernel==0:
      n_layers    = n_layers - 1
      last_kernel = kernel

    # Simple CNN
    inputs  = tf.keras.Input(shape=input_shape)
    x       = tf.keras.layers.Conv3D( 8,kernel,padding='same', activation='relu',kernel_initializer='he_uniform')(inputs)
    #x       = tf.keras.layers.Conv3D(16,kernel,padding='same', activation='relu',kernel_initializer='he_uniform')(x)

    # Layers to convolute down from dimension Np1^3 to last_kernel^3
    for i in range(n_layers):
      nf    = max( (8/(np.power(2,i))), 2) # number of filters
      x     = tf.keras.layers.Conv3D(nf,kernel,padding='valid',activation='relu',kernel_initializer='he_uniform')(x)

    x       = tf.keras.layers.Conv3D( 1,last_kernel,padding='valid',activation= 'sigmoid', kernel_initializer='he_uniform')(x)
    x       = tf.keras.layers.Lambda(lambda x: x*0.5)(x)
    outputs = tf.keras.layers.Flatten()(x)

    # Create keras model
    self._model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Actor_CNN')
    if debug > 0:
      self._model.summary()


class ValueNetCNN(network.Network):
  """
  Value network architecture purely built from 3D convolutional layers
  and one fully-connected layer to 'average' over all elements.
  ATTENTION: Might be incomparible for certain choices for the kernel size and
             the polyinomial degree N of FLEXI!
  """

  def __init__(self
              ,input_tensor_spec
              ,kernel=3
              ,debug=0):
    super(ValueNetCNN, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name='ValueNetCNN')
    #self._output_tensor_spec = output_tensor_spec
    self._input_tensor_spec = input_tensor_spec

    # Build model
    self.kernel=kernel
    self._buildmodel(self.kernel,debug)

  def call(self, observations, step_type, network_state):
    del step_type

    # We extend the input tensor by a time dimension, if it has none
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    if outer_rank == 1: # only batch_size
      observations = tf.expand_dims(observations, axis=1)

    # Evaluate model
    value = self._model(observations)

    # Squash added time dimension if added above
    if outer_rank == 1: # only batch_size
      value = tf.squeeze(value, axis=1)

    # Remove last output dimension, which is of size [1] for the value net
    # since the TF PPO agents seems to expect only a [batch x time] tensor.
    value = tf.squeeze(value, axis=-1)

    return value, network_state

  def _buildmodel(self,kernel,debug):
    # Get shape of input tensor
    shape  = self._input_tensor_spec.shape
    nVar   = shape[-1]
    Np1    = shape[-2]
    nElems = shape[-5]

    # Determine network architecture, which results in scalar output for given kernelsize and N
    n_layers    = int(np.floor((Np1-1)/(kernel-1))) # Times to apply standard kernelsize
    last_kernel = Np1 - n_layers * (kernel-1)  # "Remainder": kernelsize for last layer
    # If last_kernel is zero, set it to standard kernelsize and reduce n_layers by one
    if last_kernel==0:
      n_layers    = n_layers - 1
      last_kernel = kernel

    # Simple CNN
    inputs  = tf.keras.Input(shape=(None,nElems,Np1,Np1,Np1,nVar))
    x       = tf.keras.layers.Reshape((-1,Np1,Np1,Np1,nVar))(inputs)
    x       = tf.keras.layers.Conv3D( 8,kernel,padding='same' ,activation='relu',kernel_initializer='he_uniform')(x)
    #x       = tf.keras.layers.Conv3D(16,kernel,padding='same' ,activation='relu',kernel_initializer='he_uniform')(x)

    # Layers to convolute down from dimension Np1^3 to last_kernel^3
    for i in range(n_layers):
      nf    = max( (8/(np.power(2,i))), 2) # number of filters
      x     = tf.keras.layers.Conv3D(nf,kernel,padding='valid',activation='relu',kernel_initializer='he_uniform')(x)

    x       = tf.keras.layers.Conv3D( 1,last_kernel,padding='valid',activation='relu',kernel_initializer='he_uniform')(x)
    x       = tf.keras.layers.Flatten()(x)
    x       = tf.keras.layers.Reshape((-1,nElems))(x)
    x       = tf.keras.layers.Dense( 16,activation='relu',kernel_initializer='he_uniform')(x)
    outputs = tf.keras.layers.Dense(  1,activation= None ,kernel_initializer='he_uniform')(x)

    # Create keras model
    self._model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Value_CNN')
    if debug > 0:
      self._model.summary()


class ActionNetDG(network.Network):
  """
  Policy network architecture built from 3D convolutional layers. Average Pooling layers
  ensure that a network of this architecture can be applied to or trained on DG data of
  any polynomial degree. This allows to train a single DG-Net for a variety of N.
  """

  def __init__(self
              ,input_tensor_spec
              ,output_tensor_spec
              ,kernel=3
              ,action_std=0.02
              ,debug=0):
    super(ActionNetDG, self).__init__(
        input_tensor_spec = input_tensor_spec,
        state_spec=(),
        name='ActionNetDG')
    self._output_tensor_spec = output_tensor_spec
    self._action_std = action_std

    # Build model
    self.kernel=kernel
    self._buildmodel(kernel,debug)

  def call(self, observations, step_type, network_state):
    del step_type

    # We use batch_squash here in case the observations have a time sequence compoment.
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    batch_squash = utils.BatchSquash(outer_rank)
    observations = tf.nest.map_structure(batch_squash.flatten, observations)

    # Evaluate model
    action_means = self._model(observations)

    # Unsquash in time
    action_means = batch_squash.unflatten(action_means)

    # Scale to specs
    #action_means = tanh_squash_to_spec(action_means, self._output_tensor_spec)

    # Get standard deviation for actions
    action_std = tf.ones_like(action_means)*self._action_std

    # Return distribution
    return tfp.distributions.MultivariateNormalDiag(action_means, action_std), network_state

  def _buildmodel(self,kernel,debug):
    input_shape = self._input_tensor_spec.shape

    # DGnet
    inputs  = tf.keras.Input(shape=(None,None,None,None,input_shape[-1]))
    x       = tf.keras.layers.Conv3D( 8,kernel,padding='same' , activation='relu',kernel_initializer='he_uniform')(inputs)
    x       = tf.keras.layers.Conv3D(16,kernel,padding='same' , activation='relu',kernel_initializer='he_uniform')(x)
    x       = tf.keras.layers.Conv3D( 8,kernel,padding='same' , activation='relu',kernel_initializer='he_uniform')(x)
    x       = tf.keras.layers.Conv3D( 4,kernel,padding='same' , activation='relu',kernel_initializer='he_uniform')(x)
    x       = tf.keras.layers.Conv3D( 1,kernel,padding='valid', activation= None ,kernel_initializer='he_uniform')(x)
    x       = tf.keras.layers.Activation('sigmoid')(x)
    x       = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling3D())(x) # Apply along the element dimension
    x       = tf.keras.layers.Lambda(lambda x: x*0.5)(x)
    outputs = tf.keras.layers.Flatten()(x)

    # Create keras model
    self._model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Actor_DG')
    if debug > 0:
      self._model.summary()


class ValueNetDG(network.Network):
  """
  Value network architecture built from 3D convolutional layers. Average Pooling layers
  ensure that a network of this architecture can be applied to or trained on DG data of
  any polynomial degree. This allows to train a single DG-Net for a variety of N.
  """
  def __init__(self, input_tensor_spec,kernel=3,debug=0):
    super(ValueNetDG, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name='ValueNetDG')
    #self._output_tensor_spec = output_tensor_spec
    self._input_tensor_spec = input_tensor_spec

    # Build model
    self.kernel=kernel
    self._buildmodel(self.kernel,debug)

  def call(self, observations, step_type, network_state):
    del step_type

    # We use batch_squash here in case the observations have a time sequence compoment.
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    batch_squash = utils.BatchSquash(outer_rank)
    observations = tf.nest.map_structure(batch_squash.flatten, observations)

    # Evaluate model
    value = self._model(observations)

    # Unsquash in time
    value = batch_squash.unflatten(value)

    # Remove last output dimension, which is of size [1] for the value net
    # since the TF PPO agents seems to expect only a [batch x time] tensor.
    value = tf.squeeze(value, axis=-1)

    return value, network_state

  def _buildmodel(self,kernel,debug):
    # Get shape of input tensor
    input_shape  = self._input_tensor_spec.shape

    # DGNet - Value
    inputs  = tf.keras.Input(shape=(None,None,None,None,input_shape[-1]))
    x       = tf.keras.layers.Conv3D( 8,kernel,padding='same' , activation='relu',kernel_initializer='he_uniform')(inputs)
    x       = tf.keras.layers.Conv3D(16,kernel,padding='same' , activation='relu',kernel_initializer='he_uniform')(x)
    x       = tf.keras.layers.Conv3D( 8,kernel,padding='same' , activation='relu',kernel_initializer='he_uniform')(x)
    x       = tf.keras.layers.Conv3D( 4,kernel,padding='same' , activation='relu',kernel_initializer='he_uniform')(x)
    x       = tf.keras.layers.Conv3D( 1,kernel,padding='valid', activation= None ,kernel_initializer='he_uniform')(x)
    x       = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling3D())(x) # Apply along element dimension
    outputs = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Create keras model
    self._model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Value_DG')
    if debug > 0:
      self._model.summary()
