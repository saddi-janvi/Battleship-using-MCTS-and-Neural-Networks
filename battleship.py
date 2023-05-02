import numpy as np
from typing import List, Tuple
import random
import mcts

# The board data structure creates a board based on size input

WATER, SHIP, SUNK = 0, 1, 2
UNHIT, HIT, MISS = 0, 1 ,2

class Ship:

    def __init__(self, position: tuple, orientation: bool, size: int) -> None:
        self.position = position
        self.orientation = orientation # True: Horizontal, False: vertical
        self.size = size # Size of the ship
    
    def __str__(self) -> str:
        concat = ' ' if self.orientation else '\n'
        return concat.join('1'*self.size)

    def get_values(self):
        return (self.position, self.orientation, self.size)

class Board:

    # Initialize the game board with all empty positions
    def __init__(self, size: int) -> None:

        self.size = size # Board size
        self.grid = np.zeros((size, size), dtype=int) # Board grid
        self.hits = np.zeros((size, size), dtype=int) # Board hits
        self.ships = []

    # Returns a string representation of the board
    def __str__(self) -> str:

        spacing = ' '*3 # Spacing between the grid and hits
        grid_icon = ['.', 's', 'X'] # Icons for the grid
        hits_icon = ['_', 'X', '.'] # Icons for the hits

        # Different lines of the string
        lines = []

        # Add the titles
        header = 'BOARD'.ljust(self.size*2-1) + spacing + 'HITS'
        lines.append(header) # Appending the titles line

        # Add the grids and hits
        grid = [[grid_icon[elem] for elem in row] for row in self.grid]
        hits = [[hits_icon[elem] for elem in row] for row in self.hits]
        for row in range(self.size):
            lines.append(' '.join(grid[row])+spacing+' '.join(hits[row]))
        return '\n'.join(lines)
    
    # Returns valid positions to place a ship on the board given orientation and size
    def get_valid_ship_positions(self, orientation: bool, size: int, simulation: bool=False, grid: np.ndarray=None, max_size: int=None) -> list:
        if not simulation:
            grid = self.grid
            max_size = self.size
        valid_positions = []
        water_positions = np.argwhere(grid == WATER)
        if orientation:
            for pos in water_positions:
                x, y = pos
                if y + size < max_size:
                    locs = grid[x][y:y+size]
                    if sum(locs) == 0:
                        valid_positions.append((x,y))
        else:
            for pos in water_positions:
                x, y = pos
                if x + size < max_size:
                    locs = [row[y] for row in grid[x:x+size]]
                    if sum(locs) == 0:
                        valid_positions.append((x,y))
        return np.array(valid_positions)
    
    def add_ship(self, ship: Ship, simulate: bool=False) -> np.ndarray | bool:
        position, orientation, size = ship.get_values()
        grid = self.grid
        x, y = position

        if orientation:
            for i in range(y, y+size): grid[x][i] = SHIP
        else:
            for i in range(x, x+size): grid[i][y] = SHIP

        if not simulate:
            self.grid = grid
            self.ships.append(ship)
            return True
        return grid
    
    def get_valid_shots(self, simulate: bool=False, grid: np.ndarray=None) -> np.ndarray:
        return np.argwhere(self.grid == WATER) if not simulate else np.argwhere(grid == WATER)
    
    def get_valid_actions(self, simulate: bool=False, hits: np.ndarray=None) -> np.ndarray:
        return np.argwhere(self.hits == UNHIT) if not simulate else np.argwhere(hits == UNHIT)
    
    def fire_shot(self, position: tuple, simulate: bool=False) -> bool | np.ndarray:
        x, y = position
        grid = self.grid
        if grid[x][y] == SHIP:
            grid[x][y] = SUNK
            if simulate:
                return grid
            else: 
                self.grid = grid
                return True
        else:
            return False



class Player:
    def __init__(self, size: int) -> None:
        self.board = Board(size=size)
        self.ship_sizes = None
        self.score = 0

    def __str__(self) -> str:
        return str(self.board)
    
    def add_ship(self, ship: Ship, simulate: bool=False):
        self.board.add_ship(ship=ship, simulate=simulate)

    def fire_shot(self, position) -> bool:
        return self.board.fire_shot(position=position)

    def update_hit(self, shot: tuple, hit: bool):
        x,y = shot
        if hit: 
            self.board.hits[x][y] = HIT
            self.score += 1
        else: 
            self.board.hits[x][y] = MISS
    
    def get_shot(self, simulate: bool=False) -> tuple:
        if not simulate:
            va = self.board.get_valid_actions()
            return tuple(random.choice(va))

class Human(Player):
    def __init__(self, size) -> None:
        super().__init__(size)
        
    def __str__(self) -> str:
        return '\n'.join(['Human Player',super().__str__()])
    
    def get_shot(self):
        valid_actions = self.board.get_valid_actions()
        l = len(str(self.board.size**2-1))
        print('CHOOSE A SHOT FROM THE GRID:')
        lines = []
        pos = {}
        for i in range(self.board.size):
            line = []
            for j in range(self.board.size):
                if [i,j] in valid_actions:
                    num = str(i) + str(j)
                    line.append(num.rjust(l))
                    pos[num] = (i,j)
                else:
                    line.append(' '*l)
            lines.append(' '.join(line))

        print('\n'.join(lines))
        while True:
            num = input('ENTER POSITION: ')
            try:
                return pos[num]
            except:
                print('ENTER VALID POSITION')

class AI(Player):
    def __init__(self, size) -> None:
        super().__init__(size)

    def __str__(self) -> str:
        return super().__str__()

class Baseline(AI):
    def __init__(self, size) -> None:
        super().__init__(size)

    def __str__(self) -> str:
        return '\n'.join(['Baseline Player', super().__str__()])
    
    def get_shot(self, simulate: bool=False) -> tuple:
        if not simulate:
            va = self.board.get_valid_actions()
            return tuple(random.choice(va))

class Tree(AI):
    def __init__(self, size) -> None:
        super().__init__(size)
        self.node = None
        self.node_count = 0

    def __str__(self) -> str:
        return '\n'.join(['Tree Player', super().__str__()])
    
    def get_shot(self):
        if self.node is None:
            state = (self.board.hits, self.ship_sizes)
            self.node = mcts.Node(state)
        for i in range(10):
            self.node.rollout()
            self.node_count += 10
        best_child = self.node.children[np.argmax([child.score_estimate for child in self.node.children])]
        return np.argwhere(self.board.hits != best_child.state[0])[0]

    def update_hit(self, shot: tuple, hit: bool):
        super().update_hit(shot, hit)
        best_child = self.node.children[np.argmax([child.score_estimate for child in self.node.children])]
        x,y = shot
        if hit:
            best_child.state[0][x][y] = 1
        else:
            best_child.state[0][x][y] = 2
        best_child.children = None
        self.node = best_child

class NeuralNetwork(AI):
    def __init__(self, size) -> None:
        super().__init__(size)

    def __str__(self) -> str:
        return '\n'.join(['Tree+NN Player', super().__str__()])
    
class Game:

    def __init__(self, player1: int, player2: int, size: int) -> None:
        players = [Human, Baseline, Tree, NeuralNetwork]
        self.size = size
        self.moves = 0
        self.player1: Player = players[player1](size)
        self.player2: Player = players[player2](size)
        self.turn = True
        self.ship_sizes = None

    def __str__(self) -> str:
        return 'PLAYER 1\n' + str(self.player1) if self.turn else 'PLAYER 2\n' + str(self.player2)

    def next_turn(self):
        self.turn = not self.turn
    
    def game_over(self) -> bool:
        if len(np.argwhere(self.player1.board.grid==SHIP)) == 0 or len(np.argwhere(self.player2.board.grid==SHIP)) == 0:
            return True
        return False

    def init_ships(self, num_ships: int=3, low: int=3, high: int=6):
        self.ship_sizes = np.random.randint(low,high,num_ships)
        self.player1.ship_sizes = self.player2.ship_sizes = self.ship_sizes
        self.ships1 = []
        self.ships2 = []
        for i in range(num_ships):
            orientation1 = bool(np.random.randint(2))
            orientation2 = bool(np.random.randint(2))
            position1 = random.choice(self.player1.board.get_valid_ship_positions(orientation1, self.ship_sizes[i]))
            position2 = random.choice(self.player2.board.get_valid_ship_positions(orientation2, self.ship_sizes[i]))
            self.ships1.append(Ship(position1,orientation1,self.ship_sizes[i]))
            self.ships2.append(Ship(position2,orientation2,self.ship_sizes[i]))
            self.player1.add_ship(self.ships1[i])
            self.player2.add_ship(self.ships2[i])
        
    def play_turn(self):
        if self.turn:
            shot = self.player1.get_shot()
            hit = self.player2.fire_shot(shot)
            self.player1.update_hit(shot, hit)
        else:
            shot = self.player2.get_shot()
            hit = self.player1.fire_shot(shot)
            self.player2.update_hit(shot, hit)

if __name__ == '__main__':
    ...