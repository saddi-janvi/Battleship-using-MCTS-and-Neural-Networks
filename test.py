# Testing different instances of the game and game objects and seeing if they are working

import battleship as bs
import mcts
import numpy as np
import random

def perform_test(name):
    if name == 'ship':
        ship1 = bs.Ship((1,0), True, 5)
        print(ship1)
        ship2 = bs.Ship((2,5), False, 3)
        print(ship2)
        return True

    if name == 'board':
        board = bs.Board(10)
        print(board)
        for orient, size in zip([True, False, True], range(3,6)):
                valid_positions = board.get_valid_ship_positions(orientation=orient, size=size)
                ship = bs.Ship(position=random.choice(valid_positions), orientation=orient, size=size)
                board.add_ship(ship)
        print(board)
        valid_shots = list(tuple(x) for x in board.get_valid_shots())
        valid_actions = list(tuple(x) for x in board.get_valid_actions())
        print(valid_shots, valid_actions)
        x = board.fire_shot(random.choice(valid_shots))
        print(x)
        return True
    
    if name == 'player':

        player1 = bs.Human(8)
        player2 = bs.Baseline(8)
        print(player1, end='\n\n')
        print(player2)
        return True
    
    if name == 'game':
        # print(game)
        # print(game)
        # game.next_turn()
        # print(game)
        # print(game.ship_sizes, game.ships1, game.ships2)
        p1 = p2 = 0
        for i in range(100):
            game = bs.Game(1,2,8)
            game.init_ships(num_ships=3)
            while True:
                # print(game)
                if game.game_over():
                    print('Game Over')
                    # print(game)
                    if game.turn:
                        p2 += 1
                    else:
                        p1 += 1
                    game.next_turn()
                    print(game)
                    # print(game.player2.node.state[0])
                    print(p1,p2)
                    break
                game.play_turn()
                game.next_turn()
        print(p1,p2)
        return True

    if name == 'mcts':
        print('Testing Node __init__')
        game = bs.Game(1,2,10)
        game.init_ships()
        # print(game)
        for i in range(50):
            game.play_turn()
            game.next_turn()
        hits = game.player1.board.hits
        sizes = game.ship_sizes
        print(game.player2.board.grid)
        print(hits)
        root = mcts.Node((hits, sizes))
        
        # print(root.state)
        # print(root.is_leaf())
        while True:
            if game.game_over():
                print('Game Over\nWINNER IS:')
                game.next_turn()
                print(game)
                if mcts.Node((game.player1.board.hits, game.ship_sizes)).is_leaf():
                    print('PLAYER 1 WINS AND IS LEAF')
                else:
                    print('PLAYER 2 WINS SINCE PLAYER 1 IS NOT LEAF')
                break
            game.play_turn()
            game.next_turn()
        # print(root)
        # print(root.state)
        # print(root.children)
        # root.expand()
        # print(root.children)

        def uct(node):
            Q = np.array(node.Q_values(), dtype=object)
            N = np.array(node.N_values())
            U = Q + np.sqrt(np.log(node.visits + 1) / (N + 1))
            return node.children[np.argmax(U)]

        def rollout(node: mcts.Node):
            if node.is_leaf():
                result = node.score()
            elif node.children is None:
                node.expand()
                result = node.score()
            else:
                result = rollout(node.uct())
            node.visits += 1
            node.score_total += result
            node.score_estimate = node.score_total / node.visits
            return result

        print(root.state)

        for i in range(1000):
            root.rollout()
            if i % 100 == 0:
                print(i, root.score_total, root.visits, root.score_estimate)
                if root.children is not None:
                    print([child.score_estimate for child in root.children])

        print(np.argwhere(root.state[0] != root.uct().state[0])[0])
  

        return True

if __name__ == '__main__':
    tests = [
        # 'ship',
        # 'board',
        # 'player',
        'game',
        'mcts',
    ]
    for test in tests:
        print(f'Testing {test}\n')
        response = perform_test(test)
        if response: print('\nSuccess\n')
        else: print('\nFailed\n')