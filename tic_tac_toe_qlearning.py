import random
import pickle

BOARD_SIZE = 3
EMPTY = " "
PLAYER_X = "X"
PLAYER_O = "O"

class TicTacToe:
    def __init__(self):
        self.board = [EMPTY] * (BOARD_SIZE * BOARD_SIZE)

    def reset(self):
        self.board = [EMPTY] * (BOARD_SIZE * BOARD_SIZE)

    def available_moves(self):
        return [i for i in range(9) if self.board[i] == EMPTY]

    def make_move(self, index, player):
        if self.board[index] == EMPTY:
            self.board[index] = player
            return True
        return False

    def check_winner(self):
        wins = [
            [0,1,2],[3,4,5],[6,7,8],  # rows
            [0,3,6],[1,4,7],[2,5,8],  # cols
            [0,4,8],[2,4,6]           # diagonals
        ]
        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != EMPTY:
                return self.board[a]
        if EMPTY not in self.board:
            return "Draw"
        return None

    def get_state(self):
        return "".join(self.board)

class QLearningAgent:
    def __init__(self, player, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.player = player

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0] * 9
        return self.q_table[state]

    def choose_action(self, state, available_moves):
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        qs = self.get_qs(state)
        best = max([(i, qs[i]) for i in available_moves], key=lambda x: x[1])
        return best[0]

    def learn(self, old_state, action, reward, new_state, done):
        old_qs = self.get_qs(old_state)
        future_qs = self.get_qs(new_state)
        if done:
            old_qs[action] = old_qs[action] + self.alpha * (reward - old_qs[action])
        else:
            old_qs[action] = old_qs[action] + self.alpha * (reward + self.gamma * max(future_qs) - old_qs[action])

def train(agent_x, agent_o, episodes=50000):
    env = TicTacToe()
    for _ in range(episodes):
        env.reset()
        state = env.get_state()
        done = False
        history = []

        current_player = agent_x
        other_player = agent_o

        while not done:
            moves = env.available_moves()
            action = current_player.choose_action(state, moves)
            env.make_move(action, current_player.player)
            new_state = env.get_state()
            winner = env.check_winner()

            if winner == current_player.player:
                current_player.learn(state, action, 1, new_state, True)
                other_player.learn(state, action, -1, new_state, True)
                done = True
            elif winner == "Draw":
                current_player.learn(state, action, 0, new_state, True)
                other_player.learn(state, action, 0, new_state, True)
                done = True
            else:
                current_player.learn(state, action, 0, new_state, False)
                state = new_state
                current_player, other_player = other_player, current_player

def play(agent, human_starts=True):
    env = TicTacToe()
    human = PLAYER_X if human_starts else PLAYER_O
    ai = PLAYER_O if human_starts else PLAYER_X
    current = PLAYER_X

    while True:
        print_board(env.board)
        if current == human:
            move = int(input("Enter your move (0-8): "))
            if env.make_move(move, human):
                current = ai
        else:
            state = env.get_state()
            move = agent.choose_action(state, env.available_moves())
            env.make_move(move, ai)
            current = human

        winner = env.check_winner()
        if winner:
            print_board(env.board)
            print("Winner:", winner)
            break

def print_board(board):
    print("\n".join([
        "|".join(board[i:i+3]) for i in range(0, 9, 3)
    ]))
    print()

# --- Run training ---
agent_x = QLearningAgent(PLAYER_X)
agent_o = QLearningAgent(PLAYER_O)
train(agent_x, agent_o)

# Save Q-table if you want
with open("q_table.pkl", "wb") as f:
    pickle.dump(agent_x.q_table, f)

# Let human play
play(agent_x, human_starts=True)
