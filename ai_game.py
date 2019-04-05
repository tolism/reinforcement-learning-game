import retro
import cv2
import numpy as np

import random
import keras

from replay_buffer import ReplayBuffer
from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

# List of hyper-parameters and constants
DECAY_RATE = 0.99
BUFFER_SIZE = 40000
MINIBATCH_SIZE = 64
TOT_FRAME = 3000000
EPSILON_DECAY = 1000000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 0.1
NUM_ACTIONS = 6
TAU = 0.01
# Number of frames to throw into network
NUM_FRAMES = 3

# Load game's environment
# DOWNLOAD MORE GAMES


def init():
    env = retro.make(game='Airstriker-Genesis')
    # Check game's specs
    actions = env.action_space.n
    print(actions)
    print(env.observation_space.shape)
    # Q-Learning init
    Q = np.zeros([84, 84, actions])
    eta = .628
    gma = .9
    epis = 5000
    rev_list = []  # rewards per episode calculate
# Preprocess
# DOWNLOAD CV2


def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (84, 84, 1))


class DeepQ(object):
    """Constructs the desired deep q learning network"""

    def __init__(self):
        self.construct_q_network()

    def construct_q_network(self):
        # Uses the network architecture found in DeepMind paper
        self.model = Sequential()
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(NUM_ACTIONS))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))

        # Creates a target network as described in DeepMind paper
        self.target_model = Sequential()
        self.target_model.add(Convolution2D(32, 8, 8, subsample=(4, 4),
                                            input_shape=(84, 84, NUM_FRAMES)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 3, 3))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.target_model.add(Dense(NUM_ACTIONS))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
        self.target_model.set_weights(self.model.get_weights())

        print("Successfully constructed networks.")

    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        q_actions = self.model.predict(data.reshape(1, 84, 84, NUM_FRAMES), batch_size=1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, NUM_ACTIONS)
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, NUM_ACTIONS))

        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size=1)
            fut_action = self.target_model.predict(
                s2_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size=1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)

        loss = self.model.train_on_batch(s_batch, targets)

        # Print the loss every 10 iterations.
        if observation_num % 10 == 0:
            print("We had a loss equal to ", loss)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model = load_model(path)
        print("Succesfully loaded network.")

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)


def train(num_frames):
    observation_num = 0
    curr_state = convert_process_buffer()
    epsilon = INITIAL_EPSILON
    alive_frame = 0
    total_reward = 0

    while observation_num < num_frames:
        if observation_num % 1000 == 999:
            print(("Executing loop %d" % observation_num))

        # Slowly decay the learning rate
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

        initial_state = convert_process_buffer()
        process_buffer = []

        predict_movement, predict_q_value = deep_q.predict_movement(curr_state, epsilon)

        reward, done = 0, False
        for i in range(NUM_FRAMES):
            temp_observation, temp_reward, temp_done, _ = env.step(predict_movement)
            reward += temp_reward
            process_buffer.append(temp_observation)
            done = done | temp_done

        if observation_num % 10 == 0:
            print("We predicted a q value of ", predict_q_value)

        if done:
            print("Lived with maximum time ", alive_frame)
            print("Earned a total of reward equal to ", total_reward)
            env.reset()
            alive_frame = 0
            total_reward = 0

        new_state = self.convert_process_buffer()
        replay_buffer.add(initial_state, predict_movement, reward, done, new_state)
        total_reward += reward

        if replay_buffer.size() > MIN_OBSERVATION:
            s_batch, a_batch, r_batch, d_batch, s2_batch = replay_buffer.sample(
                MINIBATCH_SIZE)
            deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
            deep_q.target_train()


'''
def train(epochs=1000):
    init_network()
    for i in range(epochs):
        s = env.reset()
        # Reshape state
        s = preprocess(s)
        rAll = 0
        d = False
        j = 0
        while j < 99:
            env.render()
            j += 1
            a = np.argmax(Q[s, :] + np.random.randn(1, action_space)*(1./(i+1)))
            # Get new state & reward from environment
            s1, r, d, _ = env.step(a)
            s1 = preprocess(s1)
            # Update Q-Table with new knowledge
            Q[s, a] = Q[s, a] + eta*(r + gma*np.max(Q[s1, :]) - Q[s, a])
            rAll += r
            s = s1
            if d == True:
                break
        rev_list.append(rAll)
        env.render()
    print "Reward Sum on all episodes " + str(sum(rev_list)/epis)
    print "Final Values Q-Table"
    print Q
'''
# while True:
#     # Keras Learning
#
#     # Next is BrainDQL case
#     action = brain.getAction()
#     actionmax = np.argmax(np.array(action))
#
#     nextObservation,reward, done, info = env.step(actionmax)
#
#     if done:
#         nextObservation = env.reset()
#     nextObservation = preprocess(nextObservation)
#     brain.setPerception(nextObservation,action,reward,terminal)

env.close()


if __name__ == '__main__':
    init()
    train()
