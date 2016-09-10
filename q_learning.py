from cvxpy import *
import numpy as np
import gurobipy


def clear_market(bid_price):
    """ Solve market clearing problem with the bid price of the agent.
    :param float bid_price: The bid price of the agent.
    """
    # problem data
    n = 3
    np.random.seed(1)
    c = np.array([bid_price, 2, 3])     # cost parameters
    d = np.array([1, 1, 1])     # demand parameters
    Y = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]])  # incidence matrix
    g_min = np.array([0, 0, 0])     # minimum generation
    g_max = np.array([5, 5, 5])     # maximum generation
    f_min = np.array([-1, -1, -1])  # minimum flow
    f_max = np.array([1, 1, 1])     # maximum flow

    # construct the problem
    g = Variable(n)
    f = Variable(n)
    objective = Minimize(c.T*g)
    constraints = [g + Y*f == d, g_min <= g, g <= g_max, f_min <= f, f <= f_max]
    prob = Problem(objective, constraints)

    # solve the problem
    prob.solve(solver=GUROBI)

    if prob.status == OPTIMAL:
        # compute profit for the agent
        profit = g.value[0] * -constraints[0].dual_value[0]
        profit = profit[0,0]
    else:
        print('Non-optimal solution with bid price', bid_price)
        profit = 0  # zero profit for a bad problem

    return profit


class QLearn:
    def __init__(self, states, actions, alpha=0.2, gamma=0.9):
        """ Initialize Q-learning model.
        :param list[string] states: List of states the Agent can be in
        :param list[string] actions: List of actions the Agent can take
        :param float alpha: learning rate
        :param float gamma: discount factor
        """
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        
        self.q = {}

        # initialize the q-values uniformly
        for state in self.states:
            for action in self.actions:
                self.q[(state, action)] = 1

    def q_lookup(self, state, action):
        """ Retrieve the Q-value for a state-action pair.
        """
        return self.q[(state, action)]

    def update_q(self, state1, action, reward, state2):
        """ Update the Q-value of moving from state1 to state2 by taking a
        certain action and receiving a reward.
        """
        q_val = self.q_lookup(state1, action)

        max_q = max([self.q_lookup(state2, a) for a in self.actions])
        term = self.alpha * (reward + self.gamma * max_q - q_val)

        self.q[(state1, action)] = q_val + term

    def policy(self, state):
        """ What is the optimal action given a state.
        """
        q_values = [self.q_lookup(state, a) for a in self.actions]
        max_q_val = np.max(q_values)

        max_idx = [i for i, val in enumerate(q_values) if val == max_q_val]
        count = len(max_idx)

        if count > 1:
            # if multiple actions lead to the same outcome, select the
            # action randomly
            i = np.random.choice(max_idx)
        else:
            i = max_idx[0]

        action = self.actions[i]

        return action


class Agent():
    def __init__(self):
        self.bid = 1
        self.curr_reward = 1
        self.actions = ['decrease', 'keep', 'increase']
        self.states = ['lower', 'same', 'higher']

    def apply_action(self, action):
        """ Apply an action in the current state and observe the new state and
        reward.
        """
        if action == 'decrease':
            self.bid -= 0.01
        elif action == 'increase':
            self.bid += 0.01

        reward = clear_market(self.bid)

        if reward > self.curr_reward:
            new_state = 'higher'
        elif reward < self.curr_reward:
            new_state = 'lower'
        else:
            new_state = 'same'

        self.curr_reward = reward

        return new_state, reward

    def simulate(self, iters):
        my_ql = QLearn(self.states, self.actions)

        curr_state = 'same'

        for i in range(iters):
            # 1. Select an action using the policy
            action = my_ql.policy(curr_state)
            # 2. Apply action, 3. observe the next state, and 4. receive payoff
            next_state, reward = self.apply_action(action)
            # 5. Update q-values
            my_ql.update_q(curr_state, action, reward, next_state)
            # 6. Move to the next state
            curr_state = next_state

            msg = """ Iteration {}: bid price {}, profit {}.
            """.format(i, self.bid, self.curr_reward)
            print(msg)


def main():
    agent = Agent()
    agent.simulate(iters=10000)

main()