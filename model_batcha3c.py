import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input, MaxPooling2D, Permute
from keras.models import Model
from keras.layers.advanced_activations import PReLU, ELU

def build_network(num_actions, agent_history_length, resized_width, resized_height):
    state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])

    inputs_v = Input(shape=(agent_history_length, resized_width, resized_height,))
    #model_v  = Permute((2, 3, 1))(inputs_v)

    model_v = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(inputs_v)
    model_v = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(model_v)
    model_v = Flatten()(model_v)
    model_v = Dense(output_dim=512)(model_v)
    model_v = PReLU()(model_v)


    action_probs = Dense(name="p", output_dim=num_actions, activation='softmax')(model_v)

    state_value = Dense(name="v", output_dim=1, activation='linear')(model_v)


    value_network = Model(input=inputs_v, output=[state_value, action_probs])


    return state, value_network

