import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import gym
import random
import numpy as np

# hyper parameters
# learning related
BATCH_SIZE = 32
LR = 1e-3
TARGET_REPLACE_ITER = 100
EPISODE = 100

# net related
GAMMA = 0.9
MEMORY_CAPACITY = 100

# env
env = gym.make("CartPole-v0")
env = env.unwrapped
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
EPSION = 0.9

class Net(nn.Module):
    def __init__(self, h_dim=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, h_dim)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.advantage = nn.Linear(h_dim, N_ACTIONS)  # get the action advantage and state
        self.state_value = nn.Linear(h_dim, 1)
        # self.out = nn.Linear(h_dim, N_ACTIONS)
        # self.out.weight.data.normal_(0, 0.1)
        self.advantage.weight.data.normal_(0, 0.1)
        self.state_value.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = Variable(torch.FloatTensor(x))
        x = self.fc1(x)
        x = F.relu(x)
        advantage = self.advantage(x)
        state_value = self.state_value(x)
        action_value = advantage+state_value-torch.unsqueeze(torch.mean(advantage, axis=1),1)
        return action_value


class DuelDQN(Net):
    def __init__(self, h_dim=10):
        super(DuelDQN, self).__init__()
        self.eval_net, self.target_net = Net(h_dim), Net(h_dim)

        self.learn_step_counter = 0  # target update when step%TARGET_REPLACE_ITER==0
        self.memory_counter = 0  # storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # store memory with transitions(s,a,r,s')
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss(reduction='mean')

    def choose_action(self, x):
        # 注意输入数据调整：此处输入为state
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if not hasattr(self, 'q'):  # 记录选的 Qmax 值
            self.q = []
            self.running_q = 0
        if np.random.uniform() < EPSION:
            action_value = self.eval_net.forward(x)
            self.running_q = self.running_q * 0.99 + 0.01 * torch.max(action_value).detach().numpy()
            self.q.append(self.running_q)
            #             print(torch.argmax(action_value,axis=1))
            action = torch.argmax(action_value, axis=1).numpy()[0]  # 注意提取数据
        else:
            #             action = env.action_space.sample() # 此处的action选择和上面一致，都是序号
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition  # 超过capacity则覆盖
        self.memory_counter += 1

    def learn(self):
        # update target_net
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 直接随机一个batch的index
        # print(sample_index)

        # 取出batch
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:(N_STATES+1)].astype(int))) # 踩大坑，longtensor为int64才能被gather
        b_r = Variable(torch.FloatTensor(b_memory[:, (N_STATES + 1):(N_STATES+2)]))  # 这里不需要管外部环境的具体分数
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        # update eval_net using batch data
        q_target = torch.squeeze(b_r) + GAMMA * torch.max(self.target_net(b_s_).detach(), 1)[0]  # 返回的是最大值和最大值索引

        # 此处返回的是下一状态的最大action_value，注意不要更新target_net
        q_eval = self.eval_net(b_s).gather(1, b_a)  # 预期动作价值，batch对应状态-动作价值
        loss = self.loss_func(q_eval, torch.unsqueeze(q_target, 1))   # 优化td-target
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1

if __name__ == '__main__':
    # learning progress
    dqn = DQN()

    for i in range(EPISODE):
        s = env.reset()
        returns = 0
        while True:
            env.render()

            # perform action
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)

            # modify reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.8
            r = r1 + r2
            returns += r
            if dqn.memory_counter <= MEMORY_CAPACITY:
                dqn.store_transition(s, a, r, s_)
            else:
                dqn.store_transition(s, a, r, s_)
                dqn.learn()
            if done:
                break
        if i % 10 == 0:
            print("step: %d, reward: %.2f" % (i, returns))