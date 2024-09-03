#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Relexi, a reinforcement learning framework for training
# machine learning models in simulations on high-performance computing systems.
#
# Copyright (c) 2022-2024 Marius Kurz, Andrea Beck
#
# Relexi is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Relexi is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Relexi. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils


def get_models_list():
    """Returns a list of the names of all implemented models.

    Returns:
        list: A list of classes with a `NAME` attribute.
    """
    # Filter globals to find classes with a NAME attribute
    BASE = network.Network  # Base class for all models
    return [
        cls for cls in globals().values()
        if isinstance(cls, type) and hasattr(cls, 'NAME') and issubclass(cls, BASE)
    ]

def get_models_dict():
    """Returns a dictionary of all implemented models.

    Returns:
        dict: A dictionary with the class names as keys and the classes as values.
    """
    return {cls.NAME: cls for cls in get_models_list()}


def from_string(name: str):
    """Returns model class based on its name.

    Args:
        model (str): The name of the model class as a string.

    Returns:
        type: The class that matches the classname.

    Raises:
        ValueError: If no class with the given name is found.
    """
    # Get all implemented models
    classes = get_models_dict()

    # Iterate over the classes to find a match for the `NAME` attribute
    for cls_name, cls in classes.items():
        if getattr(cls, 'NAME') == name:
            return cls

    # If no matching class is found, raise an error
    raise ValueError(f"No model found with name '{name} found'. Available models are: {classes}.")


class ActionNetCNN(network.Network):
    """
    Policy network architecture purely built from 3D convolutional layers
    ATTENTION: Might be incomparible for certain choices for the kernel size and
               the polyinomial degree N of FLEXI!
    """

    NAME = 'cnn_actor'
    """str: The name of the model class."""

    def __init__(self
                ,input_tensor_spec
                ,output_tensor_spec
                ,kernel=3
                ,action_std=0.02
                ,dist_type='normal'
                ,debug=0):
        super(ActionNetCNN, self).__init__(
            input_tensor_spec = input_tensor_spec,
            state_spec=(),
            name='ActionNetCNN')
        self._output_tensor_spec = output_tensor_spec
        self._action_std = action_std
        self._dist_type = dist_type

        # Build model
        self.kernel=kernel
        self._buildmodel(kernel,dist_type,debug)

    def call(self, observations, step_type, network_state):
        del step_type

        # We use batch_squash here in case the observations have a time sequence compoment.
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)

        if self._dist_type=='normal':
            # Evaluate model
            action_means = self._model(observations)

            # Unsquash in time
            action_means = batch_squash.unflatten(action_means)

            # Get standard deviation for actions
            action_std = tf.ones_like(action_means)*self._action_std

            # Return distribution
            return tfp.distributions.MultivariateNormalDiag(action_means, action_std), network_state

        elif self._dist_type=='beta':
            # Evaluate model
            beta_coeffs = self._model(observations)

            # Get arrays of alpha & beta values (tf.identity copies the data to new tensor instead of referencing)
            beta_coeff1 = batch_squash.unflatten(tf.identity(beta_coeffs[:,:,0]))
            beta_coeff2 = batch_squash.unflatten(tf.identity(beta_coeffs[:,:,1]))

            # Build scalar Beta Distributions
            beta_distr = tfp.distributions.Beta(beta_coeff1,beta_coeff2,allow_nan_stats=False)

            # Build global distribution of independent elementwise Beta distributions. Only last dimension is sampling dimension
            return tfp.distributions.Independent(distribution=beta_distr,reinterpreted_batch_ndims=1), network_state

    def _buildmodel(self,kernel,dist_type,debug):
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

        if dist_type=='normal':
            x       = tf.keras.layers.Conv3D( 1,last_kernel,padding='valid',activation= 'sigmoid', kernel_initializer='he_uniform')(x)
            #x       = tf.keras.layers.Lambda(lambda x: x*0.5)(x)
            outputs = tf.keras.layers.Flatten()(x)

        elif dist_type=='beta':
            # Filter size of last conv layer defines the number of output parameters
            bias_init = tf.keras.initializers.Constant(value=1.0)
            x         = tf.keras.layers.Conv3D( 2,last_kernel, padding='valid'
                                                 ,activation='softplus',kernel_initializer='he_uniform',bias_initializer=bias_init)(x)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
            # relu in last conv layer gives values from 0 to inf.
            # Hence, add +1+eps to avoid coefficients <=1 which break TF for greedy evaluation
            outputs = tf.keras.layers.Lambda(lambda x: x+1.+tf.keras.backend.epsilon())(x)

        # Create keras model
        self._model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Actor_CNN')
        if debug > 0:
            self._model.summary()
        self._model.save('test')

    @property
    def model(self):
        """Returns the model."""
        return self._model


class ValueNetCNN(network.Network):
    """
    Value network architecture purely built from 3D convolutional layers
    and one fully-connected layer to 'average' over all elements.
    ATTENTION: Might be incomparible for certain choices for the kernel size and
               the polyinomial degree N of FLEXI!
    """

    NAME = 'cnn_critic'
    """str: The name of the model class."""

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

    @property
    def model(self):
        """Returns the model."""
        return self._model


class ActionNetDG(network.Network):
    """
    Policy network architecture built from 3D convolutional layers. Average Pooling layers
    ensure that a network of this architecture can be applied to or trained on DG data of
    any polynomial degree. This allows to train a single DG-Net for a variety of N.
    """

    NAME = 'dg_actor'
    """str: The name of the model class."""

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

    @property
    def model(self):
        """Returns the model."""
        return self._model


class ValueNetDG(network.Network):
    """
    Value network architecture built from 3D convolutional layers. Average Pooling layers
    ensure that a network of this architecture can be applied to or trained on DG data of
    any polynomial degree. This allows to train a single DG-Net for a variety of N.
    """

    NAME = 'dg_critic'
    """str: The name of the model class."""

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

    @property
    def model(self):
        """Returns the model."""
        return self._model


class ActionNetMLP(network.Network):
    """Fully-connected policy network architecture."""

    NAME = 'mlp_actor'
    """str: The name of the model class."""

    def __init__(self
                ,input_tensor_spec
                ,output_tensor_spec
                ,n_neurons=256
                ,n_hidden_layers=2
                ,action_std=0.02
                ,dist_type='normal'
                ,debug=0):
        super().__init__(
            input_tensor_spec = input_tensor_spec,
            state_spec=(),
            name='ActionNetMLP')
        self._output_tensor_spec = output_tensor_spec
        self._action_std = action_std
        self._dist_type = dist_type

        # Build model
        self._n_neurons = n_neurons
        self._n_hidden_layers = n_hidden_layers
        self._model = self._buildmodel(debug)

    def call(self, observations, step_type, network_state):
        del step_type

        # We use batch_squash here in case the observations have a time sequence compoment.
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)

        if self._dist_type=='normal':
            # Evaluate model
            action_means = self._model(observations)

            # Unsquash in time
            action_means = batch_squash.unflatten(action_means)

            # Get standard deviation for actions
            action_std = tf.ones_like(action_means)*self._action_std

            # Return distribution
            return tfp.distributions.MultivariateNormalDiag(action_means, action_std), network_state

        elif self._dist_type=='beta':
            # Evaluate model
            beta_coeffs = self._model(observations)

            # Split into arrays of `n` alpha & `n` beta values for a total for
            # the `n`-dimensionsl prediction of the policy.
            # (tf.identity copies the data to new tensor instead of referencing)
            n = self._output_tensor_spec.shape[-1]  # Final output dimension of policy
            beta_coeff1 = batch_squash.unflatten(tf.identity(beta_coeffs[:,0:  n]))
            beta_coeff2 = batch_squash.unflatten(tf.identity(beta_coeffs[:,n:2*n]))

            # Build scalar Beta Distributions
            beta_distr = tfp.distributions.Beta(beta_coeff1,beta_coeff2,allow_nan_stats=False)

            # Build global distribution of independent elementwise Beta distributions. Only last dimension is sampling dimension
            return tfp.distributions.Independent(distribution=beta_distr,reinterpreted_batch_ndims=1), network_state

    def _buildmodel(self, debug):
        input_shape = self._input_tensor_spec.shape
        n_actions = self._output_tensor_spec.shape[-1]
        # Simple CNN
        inputs  = tf.keras.Input(shape=input_shape)
        x       = inputs
        for i in range(self._n_hidden_layers):
            x   = tf.keras.layers.Dense(self._n_neurons,activation='tanh')(x)

        if self._dist_type=='normal':
            outputs  = tf.keras.layers.Dense(n_actions,activation='sigmoid')(x)
        elif self._dist_type=='beta':
            bias_init = tf.keras.initializers.Constant(value=1.0)
            x         = tf.keras.layers.Dense(2*n_actions,activation='softplus',bias_initializer=bias_init)(x)
            outputs = tf.keras.layers.Lambda(lambda x: x+1.+tf.keras.backend.epsilon())(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Policy_MLP')
        if debug > 0:
            model.summary()
        return model

    @property
    def model(self):
        """Returns the model."""
        return self._model


class ValueNetMLP(network.Network):
    """Fully-connected value network architecture."""

    NAME = 'mlp_critic'
    """str: The name of the model class."""

    def __init__(self
                ,input_tensor_spec
                ,state_spec=()
                ,n_neurons=256
                ,n_hidden_layers=2
                ,debug=0):
        super().__init__(
            input_tensor_spec = input_tensor_spec,
            state_spec=(),
            name='ValueNetMLP')
        self._input_tensor_spec = input_tensor_spec

        # Build model
        self._n_neurons = n_neurons
        self._n_hidden_layers = n_hidden_layers
        self._model = self._buildmodel(debug)

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

    def _buildmodel(self, debug):
        input_shape = self._input_tensor_spec.shape

        # Simple CNN
        inputs  = tf.keras.Input(shape=input_shape)
        x       = inputs
        for i in range(self._n_hidden_layers):
            x   = tf.keras.layers.Dense(self._n_neurons, activation='tanh')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)

        # Create keras model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Critic_MLP')
        if debug > 0:
            model.summary()
        return model

    @property
    def model(self):
        """Returns the model."""
        return self._model
