import numpy as np
from math import sqrt
import gym
import particle_env.env
from keras import backend as K
from keras.layers import Lambda, Dense, Input
from keras.models import Model
from keras.optimizers import Adam

# Modified code from https://github.com/Hyeokreal/Actor-Critic-Continuous-Keras/blob/master/a2c_continuous.py

class A2C:

    def __init__(self):
        print("in init!!!!!!!!!!!!")
        # xPos, yPos, xVel, yVel
        self.nS = 4
        # per direction
        self.nA = 2

        self.alphaActor = 0.0001
        self.alphaCritic = 0.001
        self.gamma = .99
        self.epsilon = 1.5
        self.epsilon_min = 0.05
        self.numEpisodesDecay = 35000
        self.epsilonDecay = (self.epsilon - self.epsilon_min) / self.numEpisodesDecay

        self.xActor, self.yActor, self.critic = self.buildModel()

        self.optimizer = [self.Xactor_optimizer(), self.Yactor_optimizer(), self.critic_optimizer()]

    def buildModel(self):
        state = Input(batch_shape=(None, self.nS))

        Xactor_input = Dense(10, input_dim=self.nS, activation='relu', kernel_initializer='he_uniform')(state)
        Xactor_hidden = Dense(10, activation='relu')(Xactor_input)
        Xactor_hidden2 = Dense(10, activation='relu')(Xactor_hidden)

        Xmu_0 = Dense(1, activation='tanh', kernel_initializer='he_uniform')(Xactor_hidden2)
        Xsigma_0 = Dense(1, activation='softplus', kernel_initializer='he_uniform')(Xactor_hidden2)

        Xmu = Lambda(lambda x: x * 3)(Xmu_0)
        Xsigma = Lambda(lambda x: x + 0.0001)(Xsigma_0)



        Yactor_input = Dense(10, input_dim=self.nS, activation='relu', kernel_initializer='he_uniform')(state)
        Yactor_hidden = Dense(10, activation='relu')(Yactor_input)
        Yactor_hidden2 = Dense(10, activation='relu')(Yactor_hidden)

        Ymu_0 = Dense(1, activation='tanh', kernel_initializer='he_uniform')(Yactor_hidden2)
        Ysigma_0 = Dense(1, activation='softplus', kernel_initializer='he_uniform')(Yactor_hidden2)

        Ymu = Lambda(lambda x: x * 3)(Ymu_0)
        Ysigma = Lambda(lambda x: x + 0.0001)(Ysigma_0)

        critic_input = Dense(10, input_dim=self.nS, activation='relu', kernel_initializer='he_uniform')(state)
        critic_hidden = Dense(10, activation='relu')(critic_input)
        critic_hidden2 = Dense(10, activation='relu')(critic_hidden)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(critic_hidden2)

        Xactor = Model(inputs=state, outputs=(Xmu, Xsigma))
        Yactor = Model(inputs=state, outputs=(Ymu, Ysigma))
        critic = Model(inputs=state, outputs=state_value)

        Xactor._make_predict_function()
        Yactor._make_predict_function()
        critic._make_predict_function()

        return Xactor, Yactor, critic

    def Xactor_optimizer(self):
        action = K.placeholder(shape=(None, 1))
        advantages = K.placeholder(shape=(None, 1))


        mu, sigma_sq = self.xActor.output

        pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))

        exp_v = log_pdf * advantages

        exp_v = K.sum(exp_v + 0.01 * entropy)
        actor_loss = -exp_v

        optimizer = Adam(lr=self.alphaActor)
        updates = optimizer.get_updates(self.xActor.trainable_weights, [], actor_loss)

        train = K.function([self.xActor.input, action, advantages], [], updates=updates)
        return train

    def Yactor_optimizer(self):
        action = K.placeholder(shape=(None, 1))
        advantages = K.placeholder(shape=(None, 1))


        mu, sigma_sq = self.yActor.output

        pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))

        exp_v = log_pdf * advantages

        exp_v = K.sum(exp_v + 0.01 * entropy)
        actor_loss = -exp_v

        optimizer = Adam(lr=self.alphaActor)
        updates = optimizer.get_updates(self.yActor.trainable_weights, [], actor_loss)

        train = K.function([self.yActor.input, action, advantages], [], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, 1))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.alphaCritic)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    def get_action(self, state, epsilon=-0.1):
        if epsilon == -0.1:
            epsilon = self.epsilon
        Xmu, Xsigma_sq = self.xActor.predict(np.reshape(state, [1, self.nS]))
        Xaction = Xmu + np.sqrt(Xsigma_sq) * epsilon * np.random.randn()

        Ymu, Ysigma_sq = self.yActor.predict(np.reshape(state, [1, self.nS]))
        Yaction = Ymu + np.sqrt(Ysigma_sq) * epsilon * np.random.randn()

        action = np.append(Xaction, Yaction)

        action = np.clip(action, -3, 3)

        return action

    def get_action_optimal(self, state, epsilon=-0.1):
        Xmu, Xsigma_sq = self.xActor.predict(np.reshape(state, [1, self.nS]))

        Ymu, Ysigma_sq = self.yActor.predict(np.reshape(state, [1, self.nS]))

        action = np.append(Xmu, Ymu)

        action = np.clip(action, -3, 3)

        return action

    def train_model(self, state, action, reward, next_state, done, oldC, oldX, oldY):
        target = np.zeros((1, 1))
        advantages = np.zeros((1, 1))

        value = oldC.predict(state)[0]
        next_value = oldC.predict(next_state)[0]

        if done:
            advantages[0] = reward - value
            target[0][0] = reward
        else:
            advantages[0] = reward + self.gamma * (next_value) - value
            target[0][0] = reward + self.gamma * next_value

        self.optimizer[0]([state, [[action[0]]], advantages])
        self.optimizer[1]([state, [[action[1]]], advantages])


        self.optimizer[2]([state, target])



if __name__ == "__main__":
    env = gym.make('Particle-3-4-Sparse-v0')
    goal = np.array([3, 4]) # TODO find a way to build this into env?

    print('Running PD controller towards goal {}'.format(goal))
    numIters = 0
    numEpisodes = 0
    agent = A2C()
    oldCritic = agent.critic
    oldYActor = agent.yActor
    oldXActor = agent.xActor

    while(True):
        numPrevIters = numIters
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])

        while not done:
            # if agent.render:
                # env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, agent.nS])
            numIters += 1
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon  = agent.epsilon - agent.epsilonDecay

            agent.train_model(state, action, reward, next_state, done, oldCritic, oldXActor, oldYActor)

            score += reward
            state = next_state

            if done:
                print("episode:", numEpisodes, "  score:", score, " numIters: ", numIters - numPrevIters)

        numEpisodes += 1
        if numEpisodes % 15 == 0:
            oldYActor = agent.yActor
            oldXActor = agent.xActor
            oldCritic = agent.critic
        if numEpisodes % 50 == 0:
            print("Testing:")
            state = env.reset()
            numItersTest = 0
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, 4])

            while not done:

                action = agent.get_action_optimal(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, agent.nS])
                numItersTest += 1


                score += reward
                state = next_state

                if done:
                    print("Testing results:", " score:", score, " numIters: ", numItersTest)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon = agent.epsilon - agent.epsilonDecay

        if numEpisodes > 50000:
            exit(0)