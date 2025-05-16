import tkinter as tk
from tkinter import messagebox
from tic_tac_toe_qlearning import TicTacToe, QLearningAgent, PLAYER_X, PLAYER_O, calculate_bonus

print("Starting GUI...")

class TicTacToeGUI:
    def __init__(self, root):
        print("Initializing GUI class...")
        self.root = root
        self.root.title("Tic-Tac-Toe Q-Learning AI")
        self.env = TicTacToe()
        self.agent = QLearningAgent(PLAYER_O)
        self.human = PLAYER_X
        self.ai = PLAYER_O
        self.current = PLAYER_X
        self.buttons = []
        self.stats = {
            "Human Wins": 0,
            "AI Wins": 0,
            "Draws": 0
        }
        self.ai_history = []



        # Load Q-table if available
        try:
            import pickle
            with open("q_table_o.pkl", "rb") as f:
                self.agent.q_table = pickle.load(f)
        except:
            pass

        self.status_label = tk.Label(root, text="Your turn (X)", font=("Helvetica", 14))
        self.status_label.pack()

        self.stats_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.stats_label.pack()

        frame = tk.Frame(root)
        frame.pack()

        for i in range(9):
            btn = tk.Button(frame, text="", font=("Helvetica", 20), width=5, height=2,
                            command=lambda i=i: self.on_click(i))
            btn.grid(row=i // 3, column=i % 3)
            self.buttons.append(btn)
            #print(f"Created button {i}")

    def on_click(self, index):
        if self.env.board[index] != " " or self.current != self.human:
            return

        self.env.make_move(index, self.human)
        self.update_board()

        winner = self.env.check_winner()
        if winner:
            self.end_game(winner)
            return

        self.current = self.ai
        self.root.after(500, self.ai_move)


    def ai_move(self):
        state = self.env.get_state()
        move = self.agent.choose_action(state, self.env.available_moves())
        bonus = calculate_bonus(self.env, self.ai, move, self.human)
        self.env.make_move(move, self.ai)

        self.ai_history.append((state, move, bonus))
        self.update_board()

        winner = self.env.check_winner()
        if winner:
            self.end_game(winner)
            return

        self.current = self.human
        self.status_label.config(text="Your turn (X)")


    def update_board(self):
        for i in range(9):
            self.buttons[i].config(text=self.env.board[i])

    def end_game(self, winner):
        import pickle

        if winner == "Draw":
            msg = "It's a draw!"
            self.stats["Draws"] += 1
            reward = 0
        elif winner == self.human:
            msg = "You win!"
            self.stats["Human Wins"] += 1
            reward = -1
        else:
            msg = "AI wins!"
            self.stats["AI Wins"] += 1
            reward = 1

        self.status_label.config(text=msg)
        messagebox.showinfo("Game Over", msg)

        # Train AI based on this human-played game
        final_state = self.env.get_state()
        for i, (state, action, bonus) in enumerate(reversed(self.ai_history)):
            done = (i == 0)
            self.agent.learn(state, action, reward + bonus, final_state, done)
            reward *= 0.9  # Discount future reward

        # Save updated Q-table
        with open("q_table_o.pkl", "wb") as f:
            pickle.dump(self.agent.q_table, f)

        self.stats_label.config(
            text=f"Wins: {self.stats['Human Wins']} | "
                f"Losses: {self.stats['AI Wins']} | "
                f"Draws: {self.stats['Draws']}"
        )

        self.reset_game()


    def reset_game(self):
        self.env.reset()
        for btn in self.buttons:
            btn.config(text="")
        self.current = PLAYER_X
        self.status_label.config(text="Your turn (X)")
        self.ai_history.clear()

if __name__ == "__main__":
    #print("Launching Tic-Tac-Toe GUI...")
    root = tk.Tk()
    gui = TicTacToeGUI(root)
    print("Entering mainloop...")
    root.mainloop()
