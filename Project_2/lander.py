import os

from collections import deque
import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# import ngraph_bridge
import matplotlib.pyplot as plt

# use plaid ml to force GPU usage
# ngraph bridge to force GPU usage
# ngraph_bridge.set_backend('PLAIDML')
# ngraph_bridge.enable()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

class LunarDeepQLearner:
    def __init__(self, learning_rate=0.0001, gamma=.99, steps=1000):
        self.env = gym.make('LunarLander-v2')
        self.model = Sequential
        self.target = Sequential
        self.learning_rate = learning_rate
        self.tau = 0.0001
        self.epsilon = 1
        self.episodes = 2000
        self.gamma = gamma
        self.steps = steps
        self.interval = 4
        self.batch_size = 64
        self.time = 0
        self.memory = deque(maxlen=int(5000))
        print("Init with lr: %.4f, gamma: %.2f, steps: %d" % (self.learning_rate, self.gamma, self.steps))
    def setup_env(self):
        self.env.reset()

    # simulate_random_action is just a test method selecting random actions
    def simulate_random_action(self, episode_count):
        for episode in range(episode_count):
            for t in range(1000):
                self.env.render()
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        self.env.close()

    def plot_episode_score(self, episode_arr, scores_arr, final_episode):
        agg = np.concatenate((scores_arr, episode_arr), axis=-1)
        plt.plot(agg)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.xlim(0, final_episode)
        plt.ylim(-400, 280)
        plt.title("Scores for episodes during model training")
        plt.savefig('episode_scores.png')
        plt.clf()

    def plot_lr_score(self, learning_rates_list, final_trial_set):
        y_pos = np.arange(len(learning_rates_list))
        plt.bar(y_pos, final_trial_set, align='center', alpha=0.5)
        plt.xlabel("Learning Rate α")
        plt.ylabel("Terminal Episode")
        plt.xticks(y_pos, learning_rates_list)
        plt.title("Convergence Speed With Varied α")
        plt.savefig('convergence_alpha.png')
        plt.clf()

    def plot_gamma_score(self, gamma_rates_list, final_trial_set):
        y_pos = np.arange(len(gamma_rates_list))
        plt.bar(y_pos, final_trial_set, align='center', alpha=0.5)
        plt.xlabel("γ")
        plt.ylabel("Terminal Episode")
        plt.xticks(y_pos, gamma_rates_list)
        plt.title("Convergence Speed With Varied γ")
        plt.savefig('convergence_gamma.png')
        plt.clf()

    def plot_step_score(self, step_count_list, final_trial_set):
        y_pos = np.arange(len(step_count_list))
        plt.bar(y_pos, final_trial_set, align='center', alpha=0.5)
        plt.xlabel("Step Count")
        plt.ylabel("Terminal Episode")
        plt.xticks(y_pos, step_count_list)
        plt.title("Convergence Speed With Varied Step Count")
        plt.savefig('convergence_steps.png')
        plt.clf()

    def plot_trained_scores(self, scores_arr, episode_arr):
        agg = np.concatenate((scores_arr, episode_arr), axis=-1)
        plt.plot(agg)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.xlim(0, 100)
        plt.ylim(0, 350)
        plt.title("Scores for episodes on the trained model")
        plt.savefig('episode_scores_trained.png')
        plt.clf()

    def generate_model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.env.observation_space.shape[0], activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=self.env.action_space.n))
        # model.summary()
        # compile model
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mean_squared_error')
        self.model = model
        self.target = model

    # select action according to epsilon greedy algorithm
    def epsilon_greedy_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def add_learning(self):
        self.time = (self.time + 1) % self.interval
        if self.time == 0:
            if len(self.memory) < self.batch_size:
                return
            states, targets = [], []
            samples = random.sample(self.memory, self.batch_size)
            # initial attempts to vectorize
            np_samples = np.array(samples)
            incomplete_samples = np.where(np_samples[:, 4] == False)
            for state, action, reward, new_state, complete in samples:
                if complete:
                    target = reward
                else:
                    q_new = np.amax(self.target.predict(new_state)[0])
                    target = reward + self.gamma * q_new

                target_pred = self.model.predict(state)
                target_pred[0][action] = target

                states.append(state[0])
                targets.append(target_pred[0])
            self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
            weights = self.model.get_weights()
            target_weights = self.target.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
            self.target.set_weights(target_weights)

    def train_model(self, file_name):
        epsilon_values = []
        scores = []
        scores_window = deque(maxlen=100)
        final_episode = 0
        for episode in range(self.episodes):
            state = self.env.reset().reshape(1, 8)
            score = 0
            for step in range(self.steps):
                action = self.epsilon_greedy_action(state)
                next_state, reward, complete, _ = self.env.step(action)
                score += reward
                next_state = next_state.reshape(1, 8)
                self.memory.append([state, action, reward, next_state, complete])
                self.add_learning()
                state = next_state
                if complete:
                    break

            scores.append(score)
            scores_window.append(score)
            epsilon_values.append(self.epsilon)
            self.epsilon *= 0.995
            self.epsilon = max(0.01, self.epsilon)
            if np.mean(scores_window) >= 200.0:
                final_episode = episode
                self.model.save("learner_dql_%s.h5" % file_name)
                break

        self.env.close()
        return epsilon_values, scores, final_episode

    # save complete model
    def save_model(self, file_name):
        self.model.save(file_name)

    def test_model(self):
        self.model = tf.keras.models.load_model('learner_dql_episode-625.h5')
        self.model.summary()
        scores = []
        for episode in range(100):
            score = 0
            cur_state = self.env.reset().reshape(1, 8)
            for step in range(1000):
                self.env.render()
                action = np.argmax(self.model.predict(cur_state)[0])
                new_state, reward, done, _ = self.env.step(action)
                new_state = new_state.reshape(1, 8)
                score += reward
                cur_state = new_state
                if done:
                    break
            print("Episode %d, score: %d" % (episode, score))
            scores.append(score)
        print("Scores for the last 100 episodes on the trained model: ", scores)
        self.env.close()
        return scores


if __name__ == '__main__':
    ll = LunarDeepQLearner()
    ll.setup_env()
    ll.generate_model()
    # just basic training model for initial graph
    print("Training basic model for plotting scores...")
    # epsilons, train_scores, final_episode = ll.train_model("default")
    # plot episode score
    # train_episodes = np.arange(0, final_episode)
    # ll.plot_episode_score(train_episodes, train_scores, final_episode)

    # experiment for trained convergence speed with varied learning rate α
    # learning_rate_list = [0.0005, 0.001, 0.005, 0.01, 0.015]
    # episode_set = []
    # for lr in learning_rate_list:
    #     nll = LunarDeepQLearner(learning_rate=lr)
    #     nll.generate_model()
    #     print("Training model with α: %.4f..." % lr)
    #     epsilons, train_scores, final_episode = nll.train_model("alpha_%.4f" % lr)
    #     episode_set.append(final_episode)
    # nll.plot_lr_score(learning_rate_list, episode_set)

    # testing the model
    scores = ll.test_model()
    test_episodes = np.arange(0, 100)
    ll.plot_trained_scores(scores, test_episodes)
    # experiment for trained convergence speed with varied gamma
    # gamma_list = [0.60, 0.75, 0.85, 0.95, 0.99]
    # episode_set = []
    # for gamma in gamma_list:
    #     nll = LunarDeepQLearner(gamma=gamma)
    #     nll.generate_model()
    #     print("Training model with γ: %.2f..." % gamma)
    #     epsilons, train_scores, final_episode = nll.train_model("gamma_%.2f" % gamma)
    #     episode_set.append(final_episode)
    # nll.plot_gamma_score(gamma_list, episode_set)
    # # experiment for trained convergence speed with varied step count
    # step_list = [200, 400, 600, 800, 1000]
    # episode_set = []
    # for step_count in step_list:
    #     nll = LunarDeepQLearner(steps=step_count)
    #     nll.generate_model()
    #     print("Training model with step_count: %d..." % step_count)
    #     epsilons, train_scores, final_episode = nll.train_model("step_count_%d" % step_count)
    #     episode_set.append(final_episode)
    # nll.plot_step_score(step_list, episode_set)
