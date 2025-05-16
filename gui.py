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
        self.agent_x = QLearningAgent(PLAYER_X)
        self.agent_o = QLearningAgent(PLAYER_O)

        self.human_starts_as = PLAYER_X  # toggles each game
        self.current = PLAYER_X
        self.ai = None
        self.human = None
        self.ai_agent = None

        self.buttons = []
        self.stats = {
            "Human Wins": 0,
            "AI Wins": 0,
            "Draws": 0
        }
        self.ai_history = []

        # Load Q-table if available
        import pickle
        try:
            with open("q_table_x.pkl", "rb") as f:
                self.agent_x.q_table = pickle.load(f)
        except:
            pass

        try:
            with open("q_table_o.pkl", "rb") as f:
                self.agent_o.q_table = pickle.load(f)
        except:
            pass

        self.root.geometry("500x600")  # Optional: set a larger window size
        self.root.resizable(False, False)

        # Status label
        self.status_label = tk.Label(root, text="Your turn (X)", font=("Helvetica", 18))
        self.status_label.pack(pady=10)

        # Stats label
        self.stats_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.stats_label.pack(pady=5)

        # Frame for the game board
        frame = tk.Frame(root, bg="#444")
        frame.pack(pady=20)

        # Create larger buttons with padding
        for i in range(9):
            btn = tk.Button(
                frame, text="", font=("Helvetica", 36, "bold"),
                width=4, height=2, bg="#f0f0f0", fg="#222",
                activebackground="#ddd", borderwidth=2,
                command=lambda i=i: self.on_click(i)
            )
            btn.grid(row=i // 3, column=i % 3, padx=8, pady=8)
            self.buttons.append(btn)

        # Optional: Add a Reset button
        reset_btn = tk.Button(root, text="Reset Game", font=("Helvetica", 12), command=self.reset_game)
        reset_btn.pack(pady=10)


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
        move = self.ai_agent.choose_action(state, self.env.available_moves())
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

        # Train correct agent
        final_state = self.env.get_state()
        for i, (state, action, bonus) in enumerate(reversed(self.ai_history)):
            done = (i == 0)
            self.ai_agent.learn(state, action, reward + bonus, final_state, done)
            reward *= 0.9

        # Save correct Q-table
        table_path = "q_table_x.pkl" if self.ai == PLAYER_X else "q_table_o.pkl"
        with open(table_path, "wb") as f:
            pickle.dump(self.ai_agent.q_table, f)

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
        self.ai_history.clear()

        # Toggle roles
        if self.human_starts_as == PLAYER_X:
            self.human = PLAYER_O
            self.ai = PLAYER_X
            self.ai_agent = self.agent_x
            self.current = PLAYER_X  # AI goes first
            self.status_label.config(text="AI thinking...")
            self.root.after(500, self.ai_move)
        else:
            self.human = PLAYER_X
            self.ai = PLAYER_O
            self.ai_agent = self.agent_o
            self.current = PLAYER_X  # Human goes first
            self.status_label.config(text="Your turn (X)")

        # Flip for next round
        self.human_starts_as = PLAYER_X if self.human_starts_as == PLAYER_O else PLAYER_O

if __name__ == "__main__":
    #print("Launching Tic-Tac-Toe GUI...")
    root = tk.Tk()
    gui = TicTacToeGUI(root)
    print("Entering mainloop...")
    root.mainloop()
