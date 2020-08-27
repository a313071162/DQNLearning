# -*- encoding: utf-8 -*-


"""
@File    : CardPole.py
@Time    : 2020/8/26 下午7:02
@Author  : dididididi
@Email   : 
@Software: PyCharm
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import gym

episode = 5000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon_max = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        # 创建一个构造器
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon_max:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for step, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon_max > self.epsilon_min:
            self.epsilon_max = self.epsilon_max * self.epsilon_decay


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 8

    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            # 选择动作
            action = agent.act(state)
            # 得到状态和回报
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            # 记录信息到Q表
            agent.remember(state, action, reward, next_state, done)

            # 更新状态
            state = next_state

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, episode, time, agent.epsilon_max))
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)