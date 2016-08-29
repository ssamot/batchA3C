import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input, MaxPooling2D
from keras.models import Model
from keras.layers.advanced_activations import PReLU, ELU

def build_network(num_actions, agent_history_length, resized_width, resized_height):
    with tf.device("/cpu:0"):
        state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])

        inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
        model_v = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(inputs)
        model_v = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(model_v)
        model_v = Flatten()(model_v)
        model_v = Dense(output_dim=256, activation='relu')(model_v)

        inputs2 = Input(shape=(agent_history_length, resized_width, resized_height,))

        model_p = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(inputs2)
        model_p = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(model_p)
        model_p = Flatten()(model_p)
        model_p = Dense(output_dim=256, activation='relu')(model_p)

        action_probs = Dense(name="p", output_dim=num_actions, activation='softmax')(model_p)

        state_value = Dense(name="v", output_dim=1, activation='linear')(model_v)

        policy_network = Model(input=inputs2, output=action_probs)
        value_network = Model(input=inputs, output=state_value)


    return state, value_network, policy_network
