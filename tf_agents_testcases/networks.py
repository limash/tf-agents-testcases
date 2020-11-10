#!/usr/bin/env python3
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. It is distributed
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.

from collections import OrderedDict

import tensorflow as tf
from tensorflow import keras

from tf_agents.networks import network
from tf_agents.networks import q_network, categorical_q_network
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):

        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                    padding="same", use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs, **kwargs):

        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

    # def get_config(self):
    #     config = super(ResidualUnit, self).get_config()
    #     config.update({"filters": self.filters,
    #                    "strides": self.strides,
    #                    "activation": keras.activations.serialize(self.activation)})
    #     return config


class CriticNetwork(network.Network):
    # it is resnet with 17 res units and 3 dense layers
    def __init__(self,
                 input_tensor_spec,
                 activation="relu",
                 name='CriticNetwork'):

        super(CriticNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        self._main_layers = [
            keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation)
        ]
        prev_filters = 64
        for filters in [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2:
            strides = 1 if filters == prev_filters else 2
            self._main_layers.append(ResidualUnit(filters, strides=strides))
            prev_filters = filters
        self._main_layers.append(keras.layers.GlobalAvgPool2D())
        self._main_layers.append(keras.layers.Flatten())

        self._joint_layers = [
            keras.layers.Dense(1024, activation=activation),
            keras.layers.Dense(1024, activation=activation),
            keras.layers.Dense(1, activation=activation, name="value")
        ]

    def call(self, inputs, step_type=(), network_state=(), training=False):

        observations, actions = inputs
        halite_map, scalar_features = observations

        X = halite_map
        for layer in self._main_layers:
            X = layer(X)

        joint = tf.concat([X, scalar_features, actions], 1)
        for layer in self._joint_layers:
            joint = layer(joint)

        return tf.reshape(joint, [-1]), network_state


class ActorDistributionNetwork(network.DistributionNetwork):
    """Creates an actor producing either Normal or Categorical distribution.
    """
    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 activation="relu",
                 discrete_projection_net=None,
                 continuous_projection_net=None,
                 name='ActorDistributionNetwork'):
        """Creates an instance of `ActorDistributionNetwork`.
    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      discrete_projection_net: Callable that generates a discrete projection
        network to be called with some hidden state and the outer_rank of the
        state.
      continuous_projection_net: Callable that generates a continuous projection
        network to be called with some hidden state and the outer_rank of the
        state.
      name: A string representing name of the network.
    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation.
    """
        self._main_layers = [
            keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation)
        ]
        prev_filters = 64
        for filters in [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2:
            strides = 1 if filters == prev_filters else 2
            self._main_layers.append(ResidualUnit(filters, strides=strides))
            prev_filters = filters
        self._main_layers.append(keras.layers.GlobalAvgPool2D())
        self._main_layers.append(keras.layers.Flatten())

        self._joint_layers = [
            keras.layers.Dense(1024, activation=activation),
            keras.layers.Dense(1024, activation=activation),
        ]

        def map_proj(spec):
            if tensor_spec.is_discrete(spec):
                return discrete_projection_net(spec)
            else:
                return continuous_projection_net(spec)

        projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
        output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                            projection_networks)

        super(ActorDistributionNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            output_spec=output_spec,
            name=name)

        self._projection_networks = projection_networks
        self._output_tensor_spec = output_tensor_spec

    @property
    def output_tensor_spec(self):
        return self._output_tensor_spec

    def call(self, observations, step_type=None, network_state=(), training=False, mask=None):

        halite_map, scalar_features = observations

        X = halite_map
        for layer in self._main_layers:
            X = layer(X)

        joint = tf.concat([X, scalar_features], 1)
        for layer in self._joint_layers:
            joint = layer(joint)

        state = joint
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)

        def call_projection_net(proj_net):
            distribution, _ = proj_net(state, outer_rank, training=training, mask=mask)
            return distribution

        output_actions = tf.nest.map_structure(call_projection_net, self._projection_networks)
        return output_actions, network_state


class QValueNet(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 activation="relu",
                 name='QValueNet'):

        super(QValueNet, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        action_spec = tf.nest.flatten(action_spec)[0]
        num_actions = action_spec.maximum - action_spec.minimum + 1

        self._main_layers = [
            keras.layers.Conv2D(64, 1, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation)
        ]
        prev_filters = 64
        for filters in [64] * 2 + [128] * 2:  # + [256] * 2 + [512] * 2:
            strides = 1 if filters == prev_filters else 2
            self._main_layers.append(ResidualUnit(filters, strides=strides))
            prev_filters = filters
        self._main_layers.append(keras.layers.GlobalAvgPool2D())
        self._main_layers.append(keras.layers.Flatten())

        self._joint_layers = [
            keras.layers.Dense(512, activation=activation),
            keras.layers.Dense(512, activation=activation),
            keras.layers.Dense(num_actions, activation=None, name="value")
        ]

    def call(self, observations, step_type=(), network_state=(), training=False):

        feature_maps, scalar_features = observations['feature_maps'], observations['scalar_features']

        X = feature_maps
        for layer in self._main_layers:
            X = layer(X)

        joint = tf.concat([X, scalar_features], 1)
        for layer in self._joint_layers:
            joint = layer(joint)

        # return tf.reshape(joint, [-1]), network_state
        return joint, network_state


def get_q_network_simple(env):
    fc_layer_params = (100,)
    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=fc_layer_params)
    return q_net


def get_categorical_q_network_simple(env):
    fc_layer_params = (100,)
    q_net = categorical_q_network.CategoricalQNetwork(
        env.observation_spec(),
        env.action_spec(),
        num_atoms=51,
        fc_layer_params=fc_layer_params)
    return q_net


def get_q_network_halite(env):
    # def get_resblock(input_shape):
    #     model = keras.models.Sequential()
    #     model.add(keras.layers.Conv2D(32, 1, strides=1, input_shape=input_shape,  # [5, 5, 3],
    #                                   padding="same", use_bias=False))
    #     model.add(keras.layers.BatchNormalization())
    #     model.add(keras.layers.Activation("relu"))
    #     prev_filters = 32
    #     for filters in [32] * 3:  # + [128] * 4 + [256] * 6 + [512] * 3:
    #         strides = 1 if filters == prev_filters else 2
    #         model.add(ResidualUnit(filters, strides=strides))
    #         prev_filters = filters
    #     model.add(keras.layers.GlobalAvgPool2D())
    #     model.add(keras.layers.Flatten())
    #     return model
    #
    # resblock = get_resblock(env.observation_spec()['feature_maps'].shape)
    # preprocessing_layers = OrderedDict({'feature_maps': resblock,
    #                                     'scalar_features': tf.keras.layers.Flatten()})
    preprocessing_layers = OrderedDict({'feature_maps': tf.keras.layers.Flatten(),
                                        'scalar_features': tf.keras.layers.Flatten()})
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
    fc_layer_params = (512, 512, 512)
    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=fc_layer_params)
    return q_net

def get_q_conv_network_halite(env):
    q_net = QValueNet(env.observation_spec(),
                      env.action_spec())
    return q_net


def get_categorical_q_network_halite(env):
    preprocessing_layers = OrderedDict({'feature_maps': tf.keras.layers.Flatten(),
                                        'scalar_features': tf.keras.layers.Flatten()})
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
    fc_layer_params = (1024, 1024)
    q_net = categorical_q_network.CategoricalQNetwork(
        env.observation_spec(),
        env.action_spec(),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        num_atoms=51,
        fc_layer_params=fc_layer_params)
    return q_net
