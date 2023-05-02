import numpy as np
import matplotlib.pyplot as plt
import battleship as bs

def mcts_vs_baseline():
    problem_size = range(8,13)
    results = {}
    for size in problem_size:
        results[size] = []
        for i in range(100):
            game = bs.Game(1, 2, size)
            game.init_ships()
            while True:
                if game.game_over():
                    nodes = game.player2.node_count
                    score1 = game.player1.score
                    score2 = game.player2.score
                    results[size].append((nodes, score1, score2))
                    break
                game.play_turn()
                game.next_turn()
        print(f'Finished problem size {size}')
    return results
    
def nn_vs_baseline():
    pass

results = mcts_vs_baseline()

for size in range(8,13):
    r = results[size]
    counts = [r[i][0] for i in range(100)]
    score2 = [r[i][2] for i in range(100)]
    plt.title(f'Problem Size {size} Node Counts')
    plt.hist(counts)
    plt.savefig(f'./plots/{size}_nc.png', bbox_inches='tight')
    plt.show()
    plt.title(f'Problem Size {size} Score')
    plt.hist(score2)
    plt.savefig(f'./plots/{size}_score.png', bbox_inches='tight')
    plt.show()