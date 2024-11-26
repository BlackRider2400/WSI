import math

# Reprezentacja planszy: 3x3
board = [
    ["", "", ""],
    ["", "", ""],
    ["", "", ""]
]


def check_winner(board):
    # Sprawdzanie rzędów, kolumn i przekątnych
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != "":
            return row[0]

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != "":
            return board[0][col]

    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != "":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != "":
        return board[0][2]

    return None

def is_full(board):
    for row in board:
        if "" in row:
            return False
    return True

def minimax(board, depth, is_maximizing, alpha, beta):
    winner = check_winner(board)
    if winner == "X":
        return 1  # Max wygrywa
    elif winner == "O":
        return -1  # Min wygrywa
    elif is_full(board):
        return 0  # Remis

    if is_maximizing:
        max_eval = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = "X"  # Max wykonuje ruch
                    eval = minimax(board, depth + 1, False, alpha, beta)
                    board[i][j] = ""  # Cofnięcie ruchu
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = "O"  # Min wykonuje ruch
                    eval = minimax(board, depth + 1, True, alpha, beta)
                    board[i][j] = ""  # Cofnięcie ruchu
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval

def best_move(board):
    best_val = -math.inf
    move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == "":
                board[i][j] = "X"  # Próbujemy ruch Maxa
                move_val = minimax(board, 0, False, -math.inf, math.inf)
                board[i][j] = ""  # Cofnięcie ruchu
                if move_val > best_val:
                    best_val = move_val
                    move = (i, j)
    return move

def print_board(board):
    for row in board:
        print(" | ".join([cell if cell != "" else " " for cell in row]))
        print("-" * 9)


def play_game():
    current_player = "O"  # Zaczyna gracz
    while True:
        print_board(board)
        if check_winner(board) or is_full(board):
            break

        if current_player == "X":
            row, col = best_move(board)
            board[row][col] = "X"
            current_player = "O"
        else:
            # Ruch gracza O
            row = int(input("Wprowadź wiersz (0-2): "))
            col = int(input("Wprowadź kolumnę (0-2): "))
            if board[row][col] == "":
                board[row][col] = "O"
                current_player = "X"
            else:
                print("Pole zajęte! Spróbuj ponownie.")

    print_board(board)
    winner = check_winner(board)
    if winner:
        print(f"Zwycięzca: {winner}")
    else:
        print("Remis!")

if __name__ == '__main__':
    play_game()