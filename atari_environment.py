from collections import deque

import numpy as np
import skimage.transform
from scipy.misc import imresize

#
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray





class AtariEnvironment(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size agent_history_length from which environment state
    is constructed.
    Some ideas from https://github.com/miyosuda/async_deep_reinforce
    And some others from https://github.com/matthiasplappert/keras-rl

    """

    def __init__(self, gym_env, resized_width, resized_height, agent_history_length, mode = "train", crop = False):
        self.env = gym_env
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length
        self.mode = mode
        self.crop = crop

        self.gym_actions = range(gym_env.action_space.n)
        if (gym_env.spec.id == "Pong-v0" or
            gym_env.spec.id == "Breakout-v0"):
            print "Doing workaround for actions"
            # Gym returns 6 possible actions for breakout and pong.
            # Only three are used, the rest are no-ops. This just lets us
            # pick from a simplified "LEFT", "RIGHT", "NOOP" action space.
            self.gym_actions = [1,2,3]

        # Screen buffer of size AGENT_HISTORY_LENGTH to be able
        # to build state arrays of size [1, AGENT_HISTORY_LENGTH, width, height]
        self.state_buffer = deque(maxlen=4)


    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer
        """
        # Clear the state buffer
        self.state_buffer = deque(maxlen=4)

        x_t = self.env.reset()

        # #if(self.mode == 'train'):
        # no_op = np.random.randint(31)
        # for _ in range(no_op):
        #     self.env.step(0)


        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 0)

        for i in range(self.agent_history_length):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, observation):



        x_t = rgb2gray(observation)
        #x_t = skimage.color.rgb2gray(observation)

        if(self.crop):
            # hardcoded, but it's fine - they should be

            resized_observation = skimage.transform.resize(x_t, (110, 84))
            resized_observation = resized_observation.astype(np.float32)
            # crop to fit 84x84
            x_t = resized_observation[18:102,:]

            #import matplotlib.image;matplotlib.image.imsave('name.png', x_t);import time; time.sleep((100))
            #

        else:
            x_t =  np.array(imresize(x_t, (self.resized_width, self.resized_height), interp="bilinear"), dtype="float")



        x_t/=255.0

        return x_t

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """
        #lives_before = self.env.ale.lives()
        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        #terminal = self.env.ale.game_over() or (self.mode == 'train' and lives_before != self.env.ale.lives())


        x_t1 = self.get_preprocessed_frame(x_t1)
        self.state_buffer.append(x_t1)
        s_t1 = np.array(self.state_buffer)

        return s_t1, r_t, terminal, info
