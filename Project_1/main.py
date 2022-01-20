import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


class SuttonFigures(object):
    def __init__(self):
        # instantiate a map of neighbors
        self.neighbors = {
            "A": ["B"],
            "B": ["A", "C"],
            "C": ["B", "D"],
            "D": ["C", "E"],
            "E": ["D", "F"],
            "F": ["E", "G"],
            "G": ["F"]
        }
        # instantiate a map of state vectors for each state
        self.state_vectors = {
            "B": np.array([1, 0, 0, 0, 0]).transpose(),
            "C": np.array([0, 1, 0, 0, 0]).transpose(),
            "D": np.array([0, 0, 1, 0, 0]).transpose(),
            "E": np.array([0, 0, 0, 1, 0]).transpose(),
            "F": np.array([0, 0, 0, 0, 1]).transpose(),
        }
        # instantiate a map of values for different lambda values
        self.rmse = {
            0: 0.0,
            0.1: 0.0,
            0.3: 0.0,
            0.5: 0.0,
            0.7: 0.0,
            0.9: 0.0,
            1.0: 0.0
        }
        self.rmse_2 = {}
        # lambda array for figure 3
        self.lambda_list = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.alpha_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
        self.curr_state = "D"
        self.training_set_count = 100
        self.sequence_count = 10
        self.i_q_diff = np.identity(5) - np.array([
            [0, 0.5, 0, 0, 0],
            [0.5, 0, 0.5, 0, 0],
            [0, 0.5, 0, 0.5, 0],
            [0, 0, 0.5, 0, 0.5],
            [0, 0, 0, 0.5, 0]
        ])
        self.epsilon = 0.001
        self.w_ideal = np.dot(np.linalg.inv(self.i_q_diff), np.array([0, 0, 0, 0, 0.5]).reshape(5, 1))

    # generates a random walk sequence until an "A" or "G" is reached
    def generate_seq(self):
        reward = 0
        seq = []
        seq.append(self.curr_state)
        curr_state_vector = np.array([self.state_vectors[self.curr_state]])
        while self.curr_state != "A" and self.curr_state != "G":
            next = self.neighbors[self.curr_state][random.randint(0, 1)]
            seq.append(next)
            if next != "A" and next != "G":
                curr_state_vector = np.append(curr_state_vector, [self.state_vectors[next]], axis=0)
            elif next == "G":
                reward = 1
            self.curr_state = next
        # print("SEQ:", seq)
        # print("SEQVEC:", curr_state_vector)
        # print("REWARD:", reward)
        self.curr_state = "D"
        return seq, curr_state_vector.transpose(), reward

    # generates a training_set_count amount of training sets for the algorithm
    def generate_training_sets(self):
        training_sets = []
        rewards = []
        for i in range(self.training_set_count):
            seqs = []
            seq_rewards = []
            for j in range(0, self.sequence_count):
                _, seq_vec, reward = self.generate_seq()
                seqs.append(seq_vec)
                seq_rewards.append(reward)
            training_sets.append(seqs)
            rewards.append(seq_rewards)
        # print("training sets: . ", training_sets)
        return training_sets, rewards

    # calculates rmse for different values of lambda
    def determine_lambda_rmse(self, lda, alpha):
        training_sets, rewards = self.generate_training_sets()
        for i in range(self.training_set_count):
            weight = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]]).transpose()
            while True:
                delta_w_agg = np.array([[0, 0, 0, 0, 0]]).transpose()
                w_prev = weight
                for j in range(self.sequence_count):
                    err = np.array([[0, 0, 0, 0, 0]]).transpose()
                    delta_w = err.copy()
                    for k in range((training_sets[i][j]).shape[1]):
                        err = err * lda + training_sets[i][j][:, [k]]
                        if not self.is_terminal_state(training_sets, i, j, k):
                            delta_w = delta_w + alpha * (
                                    np.dot(weight.transpose(), training_sets[i][j][:, [k + 1]]) - np.dot(weight.transpose(), training_sets[i][j][:, [k]])) * err
                        else:
                            delta_w = delta_w + alpha * (
                                    rewards[i][j] - np.dot(weight.transpose(), training_sets[i][j][:, [k]])) * err
                    delta_w_agg = delta_w_agg + delta_w
                weight = weight + delta_w_agg
                if (np.linalg.norm(w_prev - weight) <= self.epsilon):
                    break
            # print("Iteration ", i, "for λ ", lda)
            self.rmse[lda] += (np.sqrt(np.mean((weight - self.w_ideal) ** 2)) / 100)

    def is_terminal_state(self, training_sets, i,j, k):
        return k == (training_sets[i][j]).shape[1] - 1
    # calculates lambda and alpha
    def determine_alpha_dependant_rmse(self):
        for lbd in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            self.rmse_2[lbd] = {}
            for alpha in self.alpha_list:
                self.rmse_2[lbd][alpha] = 0
        training_sets, rewards = self.generate_training_sets()
        for lda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for alpha in self.alpha_list:
                for i in range(self.training_set_count):
                    weight = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]]).transpose()
                    for j in range(0, self.sequence_count):
                        err = np.array([[0, 0, 0, 0, 0]]).transpose()
                        delta_w = err.copy()
                        for k in range((training_sets[i][j]).shape[1]):
                            err = err * lda + training_sets[i][j][:, [k]]
                            if not self.is_terminal_state(training_sets, i, j, k):
                                delta_w = delta_w + alpha * (np.dot(weight.transpose(), training_sets[i][j][:, [k + 1]])
                                                             - np.dot(weight.transpose(), training_sets[i][j][:, [k]])) * err
                            else:
                                delta_w = delta_w + alpha * (
                                        rewards[i][j] - np.dot(weight.transpose(), training_sets[i][j][:, [k]])) * err
                        weight = weight + delta_w
                    # print("Iteration ", i, "for λ ", lda)
                    self.rmse_2[lda][alpha] += (np.sqrt(np.mean((weight - self.w_ideal) ** 2)) / 100)
        return self.rmse_2

    # actually runs TD lambda algorithm for different values in lambda_list
    def run_lambda_rmse(self):
        for lda in self.lambda_list:
            self.determine_lambda_rmse(lda, 0.01)
        return self.rmse

    # generates a line graph based on the RMSE values calculated for different lambda values
    def generate_figure_3(self):
        rmsedf = pd.DataFrame(list(self.rmse.items()), columns=["λ", "Error"])
        rmsedf.plot(x="λ", y="Error", linestyle='-', marker='o', legend=False, color="black")
        plt.xlim(rmsedf["λ"].min() - 0.05, rmsedf["λ"].max() + 0.05)
        plt.ylim(rmsedf["Error"].min() - 0.01, rmsedf["Error"].max() + 0.01)
        plt.ylabel("Error")
        plt.margins(x=0.5, y=0.15)
        plt.annotate("Widrow-Hoff", xy=(1.0, 0.175), xytext=(0.76, 0.175))
        plt.savefig("Figure3")
        plt.clf()

    # generates a multi-series line graph based on the RMSE values calculated for different lambda/alpha values
    def generate_figure_4(self):
        rmse_df = pd.DataFrame({'α': self.alpha_list})
        for lbd in [0, 0.3, 0.8, 1]:
            # print("Iterating for lbd ", lbd, list(self.rmse_2[lbd].items()))
            rmse_lbd_df = pd.DataFrame(list(self.rmse_2[lbd].items()), columns=["α", "λ = " + str(lbd)])
            rmse_lbd_df[["λ = " + str(lbd)]] = rmse_lbd_df[["λ = " + str(lbd)]]
            rmse_df = rmse_df.merge(rmse_lbd_df, on="α", how="left")
        plt.plot("α", "λ = 0", marker='o', data=rmse_df, color="red")
        plt.plot("α", "λ = 0.3", marker='o', data=rmse_df, color="green")
        plt.plot("α", "λ = 0.8", marker='o', data=rmse_df, color="blue")
        plt.plot("α", "λ = 1", marker='o', data=rmse_df, color="yellow")
        plt.legend()
        plt.xlabel("α")
        plt.ylabel("Error")
        plt.xlim(-0.05, 0.65)
        plt.ylim(0.05, 0.8)
        plt.margins(x=0.5, y=0.15)
        plt.savefig("Figure4")
        plt.clf()

    # generates a line graph representing the average error at the best value of alpha for lambda
    def generate_figure_5(self):
        rmse_fig_5_df = pd.DataFrame({'α': self.alpha_list})
        for lbd in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            rmse_lbd_df = pd.DataFrame(list(self.rmse_2[lbd].items()), columns=["α", "λ = " + str(lbd)])
            rmse_lbd_df[["λ = " + str(lbd)]] = rmse_lbd_df[["λ = " + str(lbd)]]
            rmse_fig_5_df = rmse_fig_5_df.merge(rmse_lbd_df, on="α", how="left")
        pd.DataFrame(rmse_fig_5_df.iloc[:, 1:].min())
        rmse_fig_5_df_final = pd.DataFrame(rmse_fig_5_df.iloc[:, 1:].min()).reset_index()
        rmse_fig_5_df_final.columns = ["λ", "Average RMSE w/ Best α"]
        rmse_fig_5_df_final[["λ"]] = rmse_fig_5_df_final["λ"].str.replace("λ = ", "").astype(float)
        rmse_fig_5_df_final.plot(x="λ", y="Average RMSE w/ Best α", linestyle='-', marker='o', legend=False,
                                 color="black")
        plt.ylim(rmse_fig_5_df_final["Average RMSE w/ Best α"].min() - 0.01,
                 rmse_fig_5_df_final["Average RMSE w/ Best α"].max() + 0.01)
        plt.ylabel("Average RMSE w/ Best α")
        plt.xlim(rmse_fig_5_df_final["λ"].min() - 0.05, rmse_fig_5_df_final["λ"].max() + 0.05)
        plt.margins(x=0.5, y=0.15)
        plt.annotate("Widrow-Hoff", xy=(1.0, 0.181), xytext=(0.76, 0.181))
        plt.savefig("Figure5")
        plt.clf()


# program entrypoint
if __name__ == "__main__":
    sf = SuttonFigures()
    # rmse = sf.run_lambda_rmse()
    # sf.generate_figure_3()
    rmse_2 = sf.determine_alpha_dependant_rmse()
    # sf.generate_figure_4()
    # sf.generate_figure_5()
