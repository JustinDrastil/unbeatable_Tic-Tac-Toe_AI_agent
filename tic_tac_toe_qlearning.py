import random
import pickle
import os
from tqdm import tqdm

BOARD_SIZE = 3
EMPTY = " "
PLAYER_X = "X"
PLAYER_O = "O"
stats = {
    "Human Wins": 0,
    "Agent X Wins": 0,
    "Agent O Wins": 0,
    "Draws": 0
}

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
        # Tactical override (bypass Q-table) for immediate threats
        board = list(state)
        wins = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]

        # 1. Win if possible
        for move in available_moves:
            temp = board[:]
            temp[move] = self.player
            for a, b, c in wins:
                if temp[a] == temp[b] == temp[c] == self.player:
                    return move

        # 2. Block opponent win
        opponent = PLAYER_O if self.player == PLAYER_X else PLAYER_X
        for move in available_moves:
            temp = board[:]
            temp[move] = opponent
            for a, b, c in wins:
                if temp[a] == temp[b] == temp[c] == opponent:
                    return move

        # 3. Otherwise use Q-values
        if random.random() < self.epsilon:
            return random.choice(available_moves)

        qs = self.get_qs(state)
        max_q = max(qs[i] for i in available_moves)
        best_moves = [i for i in available_moves if qs[i] == max_q]
        return random.choice(best_moves)

    def learn(self, old_state, action, reward, new_state, done):
        old_qs = self.get_qs(old_state)
        future_qs = self.get_qs(new_state)
        if done:
            old_qs[action] = old_qs[action] + self.alpha * (reward - old_qs[action])
        else:
            old_qs[action] = old_qs[action] + self.alpha * (reward + self.gamma * max(future_qs) - old_qs[action])

def evaluate_agents(agent_x, agent_o, games=100):
    env = TicTacToe()
    results = {"X": 0, "O": 0, "Draw": 0}

    # Temporarily turn off exploration
    old_eps_x = agent_x.epsilon
    old_eps_o = agent_o.epsilon
    agent_x.epsilon = 0
    agent_o.epsilon = 0

    for _ in range(games):
        env.reset()
        state = env.get_state()
        done = False
        current = agent_x if random.random() < 0.5 else agent_o
        other = agent_o if current == agent_x else agent_x

        while not done:
            action = current.choose_action(state, env.available_moves())
            env.make_move(action, current.player)
            winner = env.check_winner()
            state = env.get_state()
            if winner:
                results[winner] += 1
                done = True
            current, other = other, current

    # Restore exploration
    agent_x.epsilon = old_eps_x
    agent_o.epsilon = old_eps_o

    print(f"Evaluation after {games} games:")
    print(f"X Wins: {results['X']}, O Wins: {results['O']}, Draws: {results['Draw']}")
    return results

def calculate_bonus(env, ai, move, opponent):
    bonus = 0
    penalty = 0
    board = env.board[:]
    wins = [
        [0,1,2],[3,4,5],[6,7,8],  # rows
        [0,3,6],[1,4,7],[2,5,8],  # cols
        [0,4,8],[2,4,6]           # diagonals
    ]

    def count_two_in_row(symbol, board_snapshot):
        count = 0
        for combo in wins:
            line = [board_snapshot[i] for i in combo]
            if line.count(symbol) == 2 and line.count(EMPTY) == 1:
                count += 1
        return count

    # Immediate win bonus
    temp = board[:]
    temp[move] = ai
    for a, b, c in wins:
        if temp[a] == temp[b] == temp[c] == ai:
            bonus += 1.0
            break

    # --- DEFENSIVE BONUS: Blocking a win ---
    temp = board[:]
    temp[move] = opponent
    if count_two_in_row(opponent, temp) > 0:
        bonus += 0.5

    # --- OFFENSIVE BONUS: Creating a fork ---
    temp = board[:]
    temp[move] = ai
    if count_two_in_row(ai, temp) >= 2:
        bonus += 0.8

    # --- BONUS: Center control ---
    if move == 4:
        bonus += 0.1

    # --- BONUS: Corner control ---
    if move in [0, 2, 6, 8]:
        bonus += 0.2

    # === PENALTY 1: Missed block ===
    for i in range(9):
        if board[i] == EMPTY:
            board[i] = opponent
            if count_two_in_row(opponent, board) > 0:
                if i != move:
                    penalty -= 0.7  # missed a more urgent block
                    break
            board[i] = EMPTY

    # === PENALTY 2: Set up a fork opportunity for opponent ===
    temp = board[:]
    temp[move] = ai
    for i in range(9):
        if temp[i] == EMPTY:
            temp[i] = opponent
            if count_two_in_row(opponent, temp) >= 2:
                penalty -= 0.8  # enabled fork
                break
            temp[i] = EMPTY

    # === PENALTY 3: Edge-only move early on ===
    if move in [1, 3, 5, 7] and board.count(EMPTY) >= 7:
        penalty -= 0.2  # early edge move is often suboptimal

    return bonus + penalty

def train(agent_x, agent_o, episodes=50000):
    env = TicTacToe()
    win_x = win_o = draw = 0

    progress = tqdm(total=episodes, desc="Training Progress")

    for i in range(episodes):
        env.reset()
        state = env.get_state()
        done = False

        # Random starting player
        if random.random() < 0.5:
            current_player, other_player = agent_x, agent_o
        else:
            current_player, other_player = agent_o, agent_x

        while not done:
            moves = env.available_moves()
            action = current_player.choose_action(state, moves)
            env.make_move(action, current_player.player)
            new_state = env.get_state()
            winner = env.check_winner()

            bonus = calculate_bonus(env, current_player.player, action, other_player.player)

            if winner == current_player.player:
                current_player.learn(state, action, 1 + bonus, new_state, True)
                other_player.learn(state, action, -1, new_state, True)
                if current_player.player == PLAYER_X:
                    win_x += 1
                else:
                    win_o += 1
                done = True
            elif winner == "Draw":
                current_player.learn(state, action, bonus, new_state, True)
                other_player.learn(state, action, bonus, new_state, True)
                draw += 1
                done = True
            else:
                current_player.learn(state, action, bonus, new_state, False)
                state = new_state
                current_player, other_player = other_player, current_player

        # --- Epsilon decay ---
        if agent_x.epsilon > 0.01:
            agent_x.epsilon *= 0.9999
        if agent_o.epsilon > 0.01:
            agent_o.epsilon *= 0.9999

        if (i + 1) % 5000 == 0:
            print(f"Episode {i+1} — X Wins: {win_x}, O Wins: {win_o}, Draws: {draw}")

        progress.update(1)

    progress.close()

def play(agent, human_starts):
    env = TicTacToe()
    human = PLAYER_X if human_starts else PLAYER_O
    ai = PLAYER_O if human_starts else PLAYER_X
    current = PLAYER_X

    history = []  # (state, action, bonus_reward)

    while True:
        print_board(env.board)

        if current == human:
            move = int(input("Enter your move (0-8): "))
            if env.make_move(move, human):
                current = ai
        else:
            state = env.get_state()
            available = env.available_moves()
            move = agent.choose_action(state, available)

            bonus = calculate_bonus(env, ai, move, human)

            # Make move for real
            if env.make_move(move, ai):
                print(f"AI ({ai}) moved to {move}")
                history.append((state, move, bonus))
                current = human
            else:
                print(f"AI attempted invalid move: {move}")
                break  # exit if something's corrupted

        winner = env.check_winner()
        if winner:
            print_board(env.board)
            print("Winner:", winner)
            # Determine winner type
            if winner == "Draw":
                stats["Draws"] += 1
            elif winner == human:
                stats["Human Wins"] += 1
            elif winner == PLAYER_X:
                stats["Agent X Wins"] += 1
            elif winner == PLAYER_O:
                stats["Agent O Wins"] += 1

            # Show results
            print("\n=== Game Stats ===")
            print(f"Human Wins:     {stats['Human Wins']}")
            print(f"Agent X Wins:   {stats['Agent X Wins']}")
            print(f"Agent O Wins:   {stats['Agent O Wins']}")
            print(f"Draws:          {stats['Draws']}")

            # Final result reward
            result_reward = 1 if winner == ai else -1 if winner == human else 0
            final_state = env.get_state()

            # Backpropagate reward through AI move history
            reward = result_reward
            for i, (state, action, bonus) in enumerate(reversed(history)):
                done = (i == 0)
                agent.learn(state, action, reward + bonus, final_state, done)
                reward *= 0.9

            # Save updated Q-table
            filename = "q_table_x.pkl" if ai == PLAYER_X else "q_table_o.pkl"
            with open(filename, "wb") as f:
                pickle.dump(agent.q_table, f)
            print(f"Updated Q-table saved to {filename}.")
            break

def print_board(board):
    print("\n".join([
        "|".join(board[i:i+3]) for i in range(0, 9, 3)
    ]))
    print()

# --- Initialize agents ---
agent_x = QLearningAgent(PLAYER_X)
agent_o = QLearningAgent(PLAYER_O)

def save_q_tables():
    with open("q_table_x.pkl", "wb") as f:
        pickle.dump(agent_x.q_table, f)
    with open("q_table_o.pkl", "wb") as f:
        pickle.dump(agent_o.q_table, f)

def main():
    while True:
        print("\nSelect mode:")
        print("1 - Train AI (self-play)")
        print("2 - Play against AI")
        print("3 - Quit")
        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            episodes = input("How many training games? (e.g. 50000): ").strip()
            if episodes.isdigit():
                train(agent_x, agent_o, int(episodes))
                save_q_tables()
                print("Training complete and Q-tables saved.")
                eval_games = input("Evaluate performance over how many games? (e.g. 100): ").strip()
                if eval_games.isdigit():
                    evaluate_agents(agent_x, agent_o, int(eval_games))
            else:
                print("Invalid number of episodes.")
        elif choice == "2":
            while True:
                human_starts = random.choice([True, False])
                print("\nHuman starts:", human_starts)
                play(agent_o if human_starts else agent_x, human_starts=human_starts)
                again = input("Play again? (y/n): ")
                if again.lower() != "y":
                    print("\n=== Final Human Play Stats ===")
                    for k, v in stats.items():
                        print(f"{k}: {v}")
                    break
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
