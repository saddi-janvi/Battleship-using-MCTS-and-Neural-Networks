# Monte Carlo Tree Search

import numpy as np
import random
# The state looks like hits, sizes

MAX_CHILDREN = 10

class Node:
    def __init__(self, state) -> None:
        self.state = state
        self.children = None
        self.score_total = 0.
        self.visits = 0
        self.parent = None
        self.score_estimate = 0.

    def N_values(self):
        return [child.visits for child in self.children]

    def Q_values(self):
        return [child.score_total / child.visits if child.visits > 0 else 0 for child in self.children]
    
    def U_values(self):
        Q = np.array(self.Q_values(), dtype=object)
        N = np.array(self.N_values())
        U = Q + np.sqrt(2*np.log(self.visits + 1) / (N + 1))
        return U

    def uct(self):
        U = self.U_values
        return self.children[np.argmax(U)]

    def rollout(self):
        if self.is_leaf():
            result = self.score()
        elif self.children is None:
            self.expand()
            result = self.score()
        else:
            result = self.uct().rollout()
        self.visits += 1
        self.score_total += result
        self.score_estimate = self.score_total / self.visits
        return result

    def is_leaf(self):
        hits, sizes = self.state
        return (hits==1).sum() == sum(sizes) or (hits == 0).sum() == 0
    
    def expand(self):
        if self.children is None:
            self.children = list(map(Node, self.get_random_children()))
            for child in self.children:
                child.parent = self

    def valid_actions(self):
        hits, sizes = self.state
        return np.argwhere(hits == 0)

    def get_random_children(self):
        hits, sizes = self.state
        children = []
        va = self.valid_actions()
        np.random.shuffle(va)
        for action in va[:MAX_CHILDREN]:
            children.append((self.perform_action(self.state, action), sizes))
        return children

    def perform_action(self, state: tuple, action: np.ndarray):
        predict = state[0].copy()
        x,y = action
        predict[x][y] = np.random.randint(1,2)
        return predict

    def perform_smart_action(self, state: tuple, action: np.ndarray):
        predict: np.ndarray = state[0].copy()
        sizes = state[1]
        x,y = action
        prob_matrix = np.zeros((len(predict), len(predict)))
        for size in sizes:
            vp = self.get_valid_ship_positions(state, size)
            for poss in vp:
                for pos in poss:
                    row, col = pos
                    if predict[row][col] == 0:
                        prob_matrix[row][col] += 1/len(poss)

        max_val = np.max(prob_matrix)
        prob_matrix = prob_matrix/max(0.01, max_val)
        if prob_matrix[x][y] > 0.2:
            predict[x][y] = 1
        else:
            predict[x][y] = 2

        return predict

    def get_valid_ship_positions(self, state, size):
        hits, sizes = state
        find = hits.copy()
        max_size = len(hits)
        find[find==2] = 100
        water_positions = np.argwhere(hits != 2)
        valid_positions = []
        for pos in water_positions:
            x, y = pos
            if y + size < max_size:
                locs = find[x][y:y+size]
                if sum(locs) == size:
                    return []
                if sum(locs) == 0:
                    valid_positions.append(np.array([(x,col) for col in range(y,y+size)]))
            if x + size < max_size:
                locs = [row[y] for row in find[x:x+size]]
                if sum(locs) == size:
                    return []
                if sum(locs) == 0:
                    valid_positions.append(np.array([(row,x) for row in range(x,x+size)]))

        return np.array(valid_positions)
    
    def score(self):
        score = 0
        hits, sizes = self.state
        water = np.argwhere(hits == 0)
        ships = np.argwhere(hits == 1)
        miss = np.argwhere(hits == 2)
        grid_size = len(hits)
        # TODO: Have a better scoring policy
        # if self.parent is not None:
        #     old_hits = self.parent.state[0]
        #     # print('Reached here')
        #     action = np.argwhere(hits != old_hits)
        #     new_hits = np.pad(hits, 1)
        # for size in sizes:
        #     for row in range(grid_size):
        #         for col in range(grid_size - size):
        #             if (hits[row,col:col+size] == 1).all():
        #                 score += 10
        #             if (hits[col:col+size, row] == 1).all():
        #                 score += 10

        if self.is_leaf():
            score += 20

        score += (ships.sum()/(miss.sum()+1))
        return score