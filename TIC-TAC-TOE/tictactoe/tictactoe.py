import math
import copy

X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = 0
    O_count = 0
        
    for row in board:
        for cell in row:
            if cell == X:
                x_count += 1
            if cell == O:
                O_count += 1
    if x_count > O_count:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    possible_actions = set()
    
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell == EMPTY:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    i, j = action

    if not (0 <= i < 3 and 0 <= j < 3):
        raise ValueError("Action is out of bounds.")
    if board[i][j] != EMPTY:
        raise ValueError("The cell is already used.")
    
    new_board = copy.deepcopy(board)

    new_board[i][j] = player(board)

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != EMPTY:
            return row[0]
        
    # Check columns
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] and board[0][j] != EMPTY:
            return board[0][j]
        
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != EMPTY:
        return board[0][0]
    
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != EMPTY:
        return board[0][2]
            
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if winner(board) is not None:
        return True

    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_player = winner(board)  # Guardamos el resultado de winner(board)
    
    if winner_player == X:
        return 1
    elif winner_player == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    if terminal(board):
        return None

    current_player = player(board)

    if current_player == X:
        best_value = -float('inf')
        best_move = None
        for action in actions(board):
            new_board = result(board, action)
            value = minimax_value(new_board)
            if value > best_value:
                best_value = value
                best_move = action

    else:  # Player O
        best_value = float('inf')  # Comenzamos con el mejor valor posible
        best_move = None
        for action in actions(board):
            new_board = result(board, action)
            value = minimax_value(new_board)
            if value < best_value:
                best_value = value
                best_move = action
        return best_move
    return best_move


def minimax_value(board):
    """
    Returns the utility of a given board using Minimax.
    """
    # Si el juego ya terminó, obtenemos la utilidad del tablero.
    if terminal(board):
        return utility(board)
    
    # Si es el turno de X (jugador que maximiza), buscamos el valor máximo.
    if player(board) == X:
        best_value = -float('inf')  # Comenzamos con el peor valor posible.
        for action in actions(board):
            new_board = result(board, action)  # Generamos el nuevo tablero.
            value = minimax_value(new_board)  # Evaluamos el tablero resultante.
            best_value = max(best_value, value)  # Buscamos el valor máximo.
        return best_value
    
    # Si es el turno de O (jugador que minimiza), buscamos el valor mínimo.
    else:
        best_value = float('inf')  # Comenzamos con el mejor valor posible.
        for action in actions(board):
            new_board = result(board, action)  # Generamos el nuevo tablero.
            value = minimax_value(new_board)  # Evaluamos el tablero resultante.
            best_value = min(best_value, value)  # Buscamos el valor mínimo.
        return best_value

