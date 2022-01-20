import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


class Player:
    def __init__(self, name="player_x", ball=None):
        self.name = name
        self.score = 0
        self.position = 0
        self.ball = ball


class World:
    def __init__(self, player_one, player_two):
        self.player_one = player_one
        self.player_two = player_two
        self.ball = np.random.randint(2)
        self.ball_position = player_one.position
        self.rows = 2
        self.columns = 4
        self.goal_state_one = [0, 4]
        self.goal_state_two = [3, 7]

    def new(self, starting_pos=[1, 2, 5, 6]):
        random_int = np.random.choice(len(starting_pos), 2, replace=False)
        self.player_one.position = starting_pos[random_int[0]]
        self.player_two.position = starting_pos[random_int[1]]
        random_val = np.random.randint(2)
        if random_val == 0:
            self.ball = self.player_one.ball
            self.ball_position = self.player_one.position
        else:
            self.ball = self.player_two.ball
            self.ball_position = self.player_two.position

    def actions(self, player_one, player_two, action_one, action_two):
        tmp_one = self.move_player(player_one, action_one)
        tmp_two = self.move_player(player_two, action_two)

        if tmp_one != player_two.position:
            player_one.position = tmp_one
        else:
            self.ball = player_two.ball

        if tmp_two != player_one.position:
            player_two.position = tmp_two
        else:
            self.ball = player_one.ball

        if self.ball:
            self.ball_position = self.player_one.position
        else:
            self.ball_position = self.player_two.position

    def move_player(self, player, action):
        if action == 0 and player.position > 3:
            player_loc = player.position - 4
        elif action == 1 and player.position not in self.goal_state_two:
            player_loc = player.position + 1
        elif action == 2 and player.position < 4:
            player_loc = player.position + 4
        elif action == 3 and player.position not in self.goal_state_one:
            player_loc = player.position - 1
        else:
            player_loc = player.position

        return player_loc

    def observe(self, action_one, action_two):
        player_one = self.player_one
        player_two = self.player_two

        if np.random.randint(2) == 0:
            self.actions(player_one, player_two, action_one, action_two)
        else:
            self.actions(player_two, player_one, action_two, action_one)

        if self.ball_position in self.goal_state_one:
            reward_one = 100
            reward_two = -100
            complete = 1
        elif self.ball_position in self.goal_state_two:
            reward_one = -100
            reward_two = 100
            complete = 1
        else:
            reward_one = 0
            reward_two = 0
            complete = 0

        return [self.player_one.position, self.player_two.position, game.ball], reward_one, reward_two, complete


def plot_graph(err_list, index_list, name="Q-Learning", linewidth=1):
    plt.plot(index_list, err_list, linewidth=linewidth, color="black")
    axes = plt.gca()
    axes.set_ylim([0, 0.6])
    plt.title(name)
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-Value Difference')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(name)
    plt.clf()


def maxmin(q):
    m = matrix(q).trans()
    n = m.size[1]

    a = np.hstack((np.ones((m.size[0], 1)), m))
    e_mat = np.hstack((np.zeros((n, 1)), -np.eye(n)))

    a = np.vstack((a, e_mat))
    a = matrix(np.vstack((a, np.hstack((0, np.ones(n))), np.hstack((0, -np.ones(n))))))

    b = matrix(np.hstack((np.zeros(a.size[0] - 2), [1, -1])))

    c = matrix(np.hstack(([-1], np.zeros(n))))

    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    return solvers.lp(c, a, b, solver='glpk')['primal objective']


def solve_corr_eq(q1, q2):
    m = matrix(q1).trans()
    n = m.size[1]

    a = np.zeros((2 * n * (n - 1), (n * n)))
    q1 = np.array(q1)
    q2 = np.array(q2)
    row = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                a[row, i * n:(i + 1) * n] = q1[i] - q1[j]
                a[row + n * (n - 1), i:(n * n):n] = q2[:, i] - q2[:, j]
                row += 1

    a = matrix(a)

    a = np.hstack((np.ones((a.size[0], 1)), a))
    e_mat = np.hstack((np.zeros((n * n, 1)), -np.eye(n * n)))

    a = np.vstack((a, e_mat))
    a = matrix(np.vstack((a, np.hstack((0, np.ones(n * n))), np.hstack((0, -np.ones(n * n))))))

    b = matrix(np.hstack((np.zeros(a.size[0] - 2), [1, -1])))

    c = matrix(np.hstack(([-1.], -(q1 + q2).flatten())))

    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    sol = solvers.lp(c, a, b, solver='glpk')

    if sol['x'] is None:
        return 0, 0
    dist = sol['x'][1:]

    return np.matmul(q1.flatten(), dist)[0], np.matmul(q2.transpose().flatten(), dist)[0]


def update_lists(err_list, index_list, one, two, ball, action_one, action_two, q_one, prev_q_val, i):
    if [one, two, ball, action_one, action_two] == [2, 1, 1, 2, 4]:
        err_list.append(abs(q_one[2, 1, 1, 2, 4] - prev_q_val))
        index_list.append(i)

def run_foe_q(game, player_one, player_two, iterations=10 ** 6, alpha=0.9, gamma=0.9, width=8, height=8, depth=2,
              actions=5, lv=500.0):
    q_table = np.zeros([height, width, depth, actions, actions])

    game.new()
    complete = 0

    err_list = []
    index_list = []

    for i in range(iterations):

        if complete == 1:
            game.new()
            complete = 0

        one = player_one.position
        two = player_two.position
        ball = game.ball

        prev_q_val = q_table[2, 1, 1, 2, 4]

        current_Q = q_table[player_one.position, player_two.position, game.ball]
        action_one = np.random.choice(actions)
        action_two = np.random.choice(actions)

        next, reward_one, reward_two, complete = game.observe(action_one=action_one, action_two=action_two)

        prime_objective = maxmin(current_Q)

        q_table[one, two, ball, action_one, action_two] = (1 - alpha) * q_table[one, two, ball, action_one, action_two] + \
                                                    alpha * ((1 - gamma) * reward_one + gamma * prime_objective)

        update_lists(err_list, index_list, one, two, ball, action_one, action_two, q_table, prev_q_val, i)

        alpha *= np.e ** (-np.log(lv) / iterations)

    plot_graph(err_list, index_list, name="Foe-Q")
    return


def run_friend_q(game, player_one, player_two, iterations=10 ** 6, alpha=0.5, epsilon=0.2, epsilon_min=0.01, gamma=0.9,
                 width=8, height=8, depth=2,
                 actions=5, lv=200.0):
    decay = (epsilon - epsilon_min) / iterations
    q_table = np.random.random([height, width, depth, actions, actions])
    game.new()
    complete = 0

    err_list = []
    index_list = err_list.copy()

    for i in range(iterations):

        if complete == 1:
            game.new()
            complete = 0

        one = player_one.position
        two = player_two.position
        ball = game.ball

        prev_q_val = q_table[2, 1, 1, 2, 4]

        action_one = np.random.randint(actions)
        action_two = np.random.randint(actions)

        next, reward_one, reward_two, complete = game.observe(action_one=action_one, action_two=action_two)
        na, nb, nball = next

        q_table[one, two, ball, action_one, action_two] = (1 - alpha) * q_table[
            one, two, ball, action_one, action_two] + \
                                                          alpha * ((1 - gamma) * reward_one + gamma * np.max(
            q_table[na, nb, nball]))

        update_lists(err_list, index_list, one, two, ball, action_one, action_two, q_table, prev_q_val, i)

        epsilon -= decay
        alpha *= np.e ** (-np.log(lv) / iterations)

    plot_graph(err_list, index_list, name="Friend Q Learning")
    return


def run_q_learning(game, player_one, player_two, iterations=10 ** 6, alpha=1.0, epsilon=0.9, alpha_min=0.001, gamma=0.9,
                   width=8, height=8, depth=2,
                   actions=5):
    alpha_decay = (alpha - alpha_min) / iterations
    q_table = np.random.random([height, width, depth, actions])
    q_one = np.zeros([height, width, depth, actions])

    game.new()
    complete = 0

    err_list = []
    index_list = []

    for i in range(iterations):

        if complete == 1:
            game.new()

        one = player_one.position
        two = player_two.position
        ball = game.ball

        prev_q_val = q_one[2, 1, 1, 2]

        if epsilon > np.random.random():
            action_one = np.random.choice(actions)
            action_two = np.random.choice(actions)
        else:
            action_one = np.argmax(q_one[one, two, ball])
            action_two = np.random.choice(actions)

        next, reward_one, reward_two, complete = game.observe(action_one=action_one, action_two=action_two)
        na, nb, nball = next

        q_one[one, two, ball, action_one] = (1 - alpha) * q_table[one, two, ball, action_one] + \
                                            alpha * ((1 - gamma) * reward_one + gamma * np.max(q_one[na, nb, nball]))

        update_lists(err_list, index_list, one, two, ball, action_one, action_two, q_one, prev_q_val, i)

        alpha -= alpha_decay

    plot_graph(err_list, index_list, name="Q-Learner", linewidth=0.5)
    return


def run_correlated_q(game, player_one, player_two, iterations=10 ** 6, alpha=0.9, gamma=0.9, width=8, height=8, depth=2,
                     actions=5, lv=500.0):
    q_one = np.zeros([height, width, depth, actions, actions])
    q_two = np.zeros([height, width, depth, actions, actions])

    game.new()
    complete = 0

    err_list = []
    index_list = err_list.copy()

    for i in range(iterations):

        if complete == 1:
            game.new()
            complete = 0

        one = player_one.position
        two = player_two.position
        ball = game.ball

        prev_q_val = q_one[2, 1, 1, 2, 4]

        action_one = np.random.choice(actions)
        action_two = np.random.choice(actions)

        q_one_state = q_one[player_one.position, player_two.position, game.ball]
        q_two_state = q_two[player_one.position, player_two.position, game.ball]

        next, reward_one, reward_two, complete = game.observe(action_one=action_one, action_two=action_two)
        r_exp_one, r_exp_two = solve_corr_eq(q_one_state, q_two_state)

        q_one[one, two, ball, action_one, action_two] = (1 - alpha) * q_one[one, two, ball, action_one, action_two] + \
                                                        alpha * ((1 - gamma) * reward_one + gamma * r_exp_one)

        q_two[one, two, ball, action_one, action_two] = (1 - alpha) * q_two[one, two, ball, action_one, action_two] + \
                                                        alpha * ((1 - gamma) * reward_two + gamma * r_exp_two)

        update_lists(err_list, index_list, one, two, ball, action_one, action_two, q_one, prev_q_val, i)

        alpha *= np.e ** (-np.log(lv) / iterations)

    plot_graph(err_list, index_list, name="Correlated-Q")
    return


if __name__ == "__main__":
    player_one = Player(name="one", ball=0)
    player_two = Player(name="two", ball=1)
    iterations = 1000000

    game = World(player_one, player_two)

    # friend Q
    run_friend_q(game, player_one, player_two, iterations)

    # Q learning
    run_q_learning(game, player_one, player_two, iterations)

    # foe Q
    run_foe_q(game, player_one, player_two, iterations)

    # correlated Q
    run_correlated_q(game, player_one, player_two, iterations)
