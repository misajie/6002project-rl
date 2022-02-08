import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from DQN import DQN
from DDQN import DDQN
from DuelDQN import DuelDQN
# MD自闭调tensor维度和gather参数，DDQN调整好好复习
# hyper parameters
# learning related
BATCH_SIZE = 32
LR = 1e-3
TARGET_REPLACE_ITER = 100
EPISODE = 100

# net related
GAMMA = 0.9
MEMORY_CAPACITY = 2000

# env
env = gym.make("CartPole-v0")
env = env.unwrapped
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
EPSION = 0.9


def train(RL, episodes):
    model = RL()
    returns = []
    for i in range(episodes):
        s = env.reset()
        temp_return = 0
        while True:
            env.render()

            # perform action
            a = model.choose_action(s)

            s_, r, done, info = env.step(a)

            # modify reward
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.8
            # r = r1 + r2
            temp_return += r
            returns.append(temp_return)
            if model.memory_counter <= MEMORY_CAPACITY:
                model.store_transition(s, a, r, s_)
            else:
                model.store_transition(s, a, r, s_)
                model.learn()
            if done:
                break
    # return model.q
    return returns

if __name__ == '__main__':
    q_natural = train(DQN, EPISODE)
    q_double = train(DDQN, EPISODE)
    q_duel = train(DuelDQN, EPISODE)
    # 出对比图
    plt.plot(np.array(q_natural), c='r', label='natural')
    plt.plot(np.array(q_double), c='b', label='double')
    plt.plot(np.array(q_duel), c='k', label='duel')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()