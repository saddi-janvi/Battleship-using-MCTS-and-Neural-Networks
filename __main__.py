# BATTLESHIP GAME ENTRY POINT

# Battleship is a two player strategy game. The players have their own square bord where they place ships. The players can only see their own boards and make guesses by attacking the opponents 

# Defining the text prompts and game environment through user input:
#  1. The game is on a square grid with each player having their own grid
#  2. The grid size can be between 5 and 14 and is chosen at the start of the 
#     game
#     by the user
#  3. The number of ships for the game will be randomly chosen at the start of 
#     the game to be between (3,5) by the program
#  4. The user will be prompted the control for each player from the following 
#     options:
#        i. Human
#       ii. Baseline AI (Places ships evenly at random on the board)
#      iii. Tree-Based AI (Uses Monte Carlo Tree Search (MCTS) to find the next
#           best move)
#       iv. tree+NN AI (Uses a padded CNN integrated in the MCTS to choose the 
#           child)
#  5. After the game environment inputs are taken from the user we will 
#     initialize
#     the game state. If there are any human players involved their user prompts
#     will be generated.
#  6. The gameplay begins waiting for any inputs to be taken before a turn 
#     is played and then waiting to press ENTER before performning the actions

from text_screens import screens as sc
import battleship as bs
import numpy as np

def start_game():
    screen = sc()
    option = screen.print_home_page()
    environment = screen.text_based_interface(option)
    play_game(environment)

def play_game(environment):
    screen = sc()
    role1, role2, size = environment
    ground = bs.Game(player1=role1-1, player2=role2-1, size=size)
    ground.init_ships(np.random.randint(3,6))
    print('Initialized game board succesfully, press ENTER to start!')
    while True:
        # screen.clear_screen()
        if ground.game_over():
            print('GAME OVER: ', end='')
            if ground.turn: 
                print('PLAYER 2 WINS')
            else: 
                print('PLAYER 1 WINS')
            ground.turn = True
            print(ground)
            ground.turn = not ground.turn
            print(ground)
            print(f'FINAL SCORES:\nPLAYER 1: {ground.player1.score}\nPLAYER 2: {ground.player2.score}')
            break

        print(ground)
        ground.play_turn()
        input('PRESS ENTER TO CONTINUE')
        ground.next_turn()

if __name__ == '__main__':
    start_game()