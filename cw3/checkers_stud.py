#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: RafaĹ Biedrzycki
Kodu tego mogÄ uĹźywaÄ moi studenci na Äwiczeniach z przedmiotu WstÄp do Sztucznej Inteligencji.
Kod ten powstaĹ aby przyspieszyÄ i uĹatwiÄ pracÄ studentĂłw, aby mogli skupiÄ siÄ na algorytmach sztucznej inteligencji.
Kod nie jest wzorem dobrej jakoĹci programowania w Pythonie, nie jest rĂłwnieĹź wzorem programowania obiektowego, moĹźe zawieraÄ bĹÄdy.
Mam ĹwiadomoĹÄ wielu jego brakĂłw ale nie mam czasu na jego poprawianie.

Zasady gry: https://en.wikipedia.org/wiki/English_draughts (w skrĂłcie: wszyscy ruszajÄ siÄ po 1 polu. Pionki tylko w kierunku wroga, damki w dowolnym)
  z nastÄpujÄcymi modyfikacjami: a) bicie nie jest wymagane,  b) dozwolone jest tylko pojedyncze bicie (bez serii).

NaleĹźy napisaÄ funkcje "minimax_a_b_recurr", "minimax_a_b" (ktĂłra woĹa funkcjÄ rekurencyjnÄ) i funkcje "*ev_func", ktĂłra oceniajÄ stan gry

ChÄtni mogÄ ulepszaÄ mĂłj kod (trzeba oznaczyÄ komentarzem co zostaĹo zmienione), mogÄ rĂłwnieĹź dodaÄ obsĹugÄ bicia wielokrotnego i wymagania bicia. MogÄ rĂłwnieĹź wdroĹźyÄ reguĹy: https://en.wikipedia.org/wiki/Russian_draughts
"""
import numpy as np
import pygame
from copy import deepcopy

FPS = 20

MINIMAX_DEPTH = 5

WIN_WIDTH = 800
WIN_HEIGHT = 800

WON_PRIZE = 10000

MOVES_HIST_LEN = 6

BOARD_WIDTH = BOARD_HEIGHT = 8

FIELD_SIZE = WIN_WIDTH / BOARD_WIDTH
PIECE_SIZE = FIELD_SIZE / 2 - 8
MARK_THICK = 2
POS_MOVE_MARK_SIZE = PIECE_SIZE / 2

BLACK_PIECES_COL = (0, 0, 0)
WHITE_PIECES_COL = (255, 255, 255)
POSS_MOVE_MARK_COL = (255, 0, 0)
DARK_BOARD_COL = (196, 164, 132)
BRIGHT_BOARD_COL = (250, 250, 250)
KING_MARK_COL = (255, 215, 0)


# count difference between the number of pieces, king+10
def basic_ev_func(board, is_black_turn):

    h = 0
    # ToDo funkcja liczy i zwraca ocene aktualnego stanu planszy

    for i in board.board:
        for j in i:
            if j.is_black():
                h += 1
                if j.is_king():
                    h += 9
            elif j.is_white():
                h -= 1
                if j.is_king():
                    h -= 9

    # board.board[row][col].is_black() - sprawdza czy to czarny kolor figury
    # board.board[row][col].is_white() - sprawdza czy to biaĹy kolor figury
    # board.board[row][col].is_king() - sprawdza czy to damka
    # wspĂłĹrzÄdne zaczynajÄ (0,0) siÄ od lewej od gĂłry
    if board.white_won or board.black_won:
        h += WON_PRIZE

    return h if is_black_turn else -h


# nagrody jak w wersji podstawowej + nagroda za stopieĹ zwartoĹci grupy
def group_prize_ev_func(board, is_black_turn):

    h = 0
    # ToDo
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            piece = board.board[row][col]
            if not piece.is_empty():
                neighbors = 0
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for d_row, d_col in directions:
                    n_row, n_col = row + d_row, col + d_col
                    if 0 <= n_row < BOARD_HEIGHT and 0 <= n_col < BOARD_WIDTH:
                        if not board.board[n_row][n_col].is_empty() and \
                                board.board[n_row][n_col].is_black() == piece.is_black():
                            neighbors += 1
                if piece.is_black():
                    h += neighbors
                else:
                    h -= neighbors

    if board.white_won or board.black_won:
        h += WON_PRIZE

    return h if is_black_turn else -h


# za kaĹźdy pion na wĹasnej poĹowie planszy otrzymuje siÄ 5 nagrody, na poĹowie przeciwnika 7, a za kaĹźdÄ damkÄ 10.
def push_to_opp_half_ev_func(board, is_black_turn):

    # ToDo
    h = 0
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            piece = board.board[row][col]
            if piece.is_black():
                h += 7 if row >= BOARD_HEIGHT // 2 else 5
                if piece.is_king():
                    h += 10
            elif piece.is_white():
                h -= 7 if row < BOARD_HEIGHT // 2 else 5
                if piece.is_king():
                    h -= 10

    if board.white_won or board.black_won:
        h += WON_PRIZE

    return h if is_black_turn else -h


# za kaĹźdy nasz pion otrzymuje siÄ nagrodÄ w wysokoĹci: (5 + numer wiersza, na ktĂłrym stoi pion) (im jest bliĹźej wroga tym lepiej), a za kaĹźdÄ damkÄ dodtakowe: 10.
def push_forward_ev_func(board, is_black_turn):
    h = 0
    # ToDo
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            piece = board.board[row][col]
            if piece.is_black():
                h += 5 + row
                if piece.is_king():
                    h += 10
            elif piece.is_white():
                h -= 5 + (BOARD_HEIGHT - 1 - row)
                if piece.is_king():
                    h -= 10

    if board.white_won or board.black_won:
        h += WON_PRIZE

    return h if is_black_turn else -h

def minimax_a_b(board, depth, plays_as_black, ev_func):
    # ToDo
    max_eval = -np.inf
    best_index = 0
    possible_moves = board.get_possible_moves(plays_as_black)
    if len(possible_moves) == 0:
        board.white_won = plays_as_black
        board.black_won = not plays_as_black
        board.is_running = False
        return None

    a = -np.inf
    b = np.inf
    for i, move in enumerate(possible_moves):
        tmp_board = deepcopy(board)
        tmp_board.make_move(move)
        eval = minimax_a_b_recurr(tmp_board, depth - 1, plays_as_black, a, b, ev_func)
        if plays_as_black:
            if eval > max_eval:
                max_eval = eval
                best_index = i
            a = max(a, eval)
        else:
            if eval < max_eval:
                max_eval = eval
                best_index = i
            b = min(b, eval)
        if a >= b:
            break

    return possible_moves[best_index]

def minimax_a_b_recurr(board, depth, move_max, a, b, ev_func):
    # ToDo
    if depth == 0 or board.end():
        return ev_func(board, move_max)

    possible_moves = board.get_possible_moves(move_max)
    if move_max:
        max_eval = -np.inf
        for move in possible_moves:
            tmp_board = deepcopy(board)
            tmp_board.make_move(move)
            eval = minimax_a_b_recurr(tmp_board, depth - 1, False, a, b, ev_func)
            max_eval = max(max_eval, eval)
            a = max(a, eval)
            if a >= b:
                break
        return a
    else:
        min_eval = np.inf
        for move in possible_moves:
            tmp_board = deepcopy(board)
            tmp_board.make_move(move)
            eval = minimax_a_b_recurr(tmp_board, depth - 1, True, a, b, ev_func)
            min_eval = min(min_eval, eval)
            b = min(b, eval)
            if a >= b:
                break
    return b

class Move:
    def __init__(self, piece, dest_row, dest_col, captures=None):
        self.piece = piece
        self.dest_row = dest_row
        self.dest_col = dest_col
        self.captures = captures

    def __eq__(self, other):
        if other is None:
            return False
        return self.piece == other.piece and self.dest_row == other.dest_row and self.dest_col == other.dest_col and self.captures == other.captures

    def __str__(self):
        return "Move from r, c:" + str(self.piece.row) + ", " + str(self.piece.col) + ", to:" + str(
            self.dest_row) + ", " + str(self.dest_col) + ", " + str(id(self.piece))


class Field:
    def is_empty(self):
        return True

    def is_white(self):
        return False

    def is_black(self):
        return False

    def __str__(self):
        return "."


class Pawn(Field):
    def __init__(self, is_white, row, col):
        self.__is_white = is_white
        self.row = row
        self.col = col

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result

    def __str__(self):
        if self.is_white():
            return "w"
        return "b"

    def is_king(self):
        return False

    def is_empty(self):
        return False

    def is_white(self):
        return self.__is_white

    def is_black(self):
        return not self.__is_white


class King(Pawn):
    def __init__(self, pawn):
        super().__init__(pawn.is_white(), pawn.row, pawn.col)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result

    def is_king(self):
        return True

    def __str__(self):
        if self.is_white():
            return "W"
        return "B"


class Board:
    def __init__(self):  # row, col
        self.board = []
        self.white_turn = True
        self.white_fig_left = 12
        self.black_fig_left = 12
        self.black_won = False
        self.white_won = False
        self.capture_exists = False
        self.last_white_mov_indx = 0
        self.white_moves_hist = [None] * MOVES_HIST_LEN
        self.black_moves_hist = [None] * MOVES_HIST_LEN
        self.last_black_mov_indx = 0
        self.black_repeats = False
        self.white_repeats = False

        self.__set_pieces()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        result.board = deepcopy(self.board)
        return result

    def __str__(self):
        to_ret = ""
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                to_ret += str(self.board[row][col])
            to_ret += "\n"
        return to_ret

    def set(self, b):
        self.white_fig_left = 0
        self.black_fig_left = 0
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                fig = Field()
                if b[row][col] == "b" or b[row][col] == "w":
                    fig = Pawn(b[row][col] == "w", row, col)

                if b[row][col] == "B" or b[row][col] == "W":
                    fig = King(Pawn(b[row][col] == "W", row, col))

                self.board[row][col] = fig
                if self.board[row][col].is_black():
                    self.black_fig_left += 1
                if self.board[row][col].is_white():
                    self.white_fig_left += 1

    def __set_pieces(self):
        for row in range(BOARD_HEIGHT):
            self.board.append([])
            for col in range(BOARD_WIDTH):
                self.board[row].append(Field())

        for row in range(BOARD_HEIGHT // 2 - 1):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(False, row, col)

        for row in range(BOARD_HEIGHT // 2 + 1, BOARD_HEIGHT):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(True, row, col)

    def get_piece_moves(self, piece):
        pos_moves = []
        row = piece.row
        col = piece.col
        if piece.is_black():
            enemy_is_white = True
        else:
            enemy_is_white = False

        if piece.is_white() or (piece.is_black() and piece.is_king()):
            dir_y = -1
            if row > 0:
                new_row = row + dir_y
                if col > 0:
                    new_col = col - 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # captures
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y >= 0 and new_col - 1 >= 0 and \
                            self.board[new_row + dir_y][new_col - 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col - 1, self.board[new_row][new_col]))
                        self.capture_exists = True

                if col < BOARD_WIDTH - 1:
                    new_col = col + 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # captures
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y >= 0 and new_col + 1 < BOARD_WIDTH and \
                            self.board[new_row + dir_y][new_col + 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col + 1, self.board[new_row][new_col]))
                        self.capture_exists = True

        if piece.is_black() or (piece.is_white() and self.board[row][col].is_king()):
            dir_y = 1
            if row < BOARD_WIDTH - 1:
                new_row = row + dir_y
                if col > 0:
                    new_col = col - 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y < BOARD_WIDTH and new_col - 1 >= 0 and \
                            self.board[new_row + dir_y][new_col - 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col - 1, self.board[new_row][new_col]))
                        self.capture_exists = True

                if col < BOARD_WIDTH - 1:
                    new_col = col + 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # captures
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y < BOARD_WIDTH and new_col + 1 < BOARD_WIDTH and \
                            self.board[new_row + dir_y][new_col + 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col + 1, self.board[new_row][new_col]))
                        self.capture_exists = True
        return pos_moves


    def get_possible_moves(self, is_black_turn):
        pos_moves = []
        self.capture_exists = False
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if (is_black_turn and self.board[row][col].is_black()) or (
                            not is_black_turn and self.board[row][col].is_white()):
                        pos_moves.extend(self.get_piece_moves(self.board[row][col]))
        return pos_moves


    def end(self):

        if self.black_repeats and self.white_repeats:

            ev = basic_ev_func(self, not self.white_turn)
            if ev > 0:
                self.black_won = True
            elif ev < 0:
                self.white_won = True
            else:
                self.black_won = True
                self.white_won = True
            return True
        return False


    def register_move(self, move):
        move_tuple = (move.piece.row, move.piece.col, move.dest_row, move.dest_col, id(move.piece))

        if self.white_turn:
            self.white_repeats = False
            if move_tuple in self.white_moves_hist:
                self.white_repeats = True
            self.white_moves_hist[self.last_white_mov_indx] = move_tuple
            self.last_white_mov_indx += 1
            if self.last_white_mov_indx >= MOVES_HIST_LEN:
                self.last_white_mov_indx = 0
        else:
            self.black_repeats = False
            if move_tuple in self.black_moves_hist:
                self.black_repeats = True
            self.black_moves_hist[self.last_black_mov_indx] = move_tuple
            self.last_black_mov_indx += 1
            if self.last_black_mov_indx >= MOVES_HIST_LEN:
                self.last_black_mov_indx = 0

    # execute move on board
    def make_move(self, move):
        d_row = move.dest_row
        d_col = move.dest_col
        row_from = move.piece.row
        col_from = move.piece.col

        self.board[d_row][d_col] = self.board[row_from][col_from]
        self.board[d_row][d_col].row = d_row
        self.board[d_row][d_col].col = d_col
        self.board[row_from][col_from] = Field()

        if move.captures:
            fig_to_del = move.captures
            self.board[fig_to_del.row][fig_to_del.col] = Field()
            if self.white_turn:
                self.black_fig_left -= 1
            else:
                self.white_fig_left -= 1

        if self.white_turn and d_row == 0 and not self.board[d_row][d_col].is_king():  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        if not self.white_turn and d_row == BOARD_WIDTH - 1 and not self.board[d_row][d_col].is_king():  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        self.white_turn = not self.white_turn


class Game:
    def __init__(self, window, board):
        self.window = window
        self.board = board
        self.something_is_marked = False
        self.marked_col = None
        self.marked_row = None
        self.pos_moves = {}

    def __draw(self):
        self.window.fill(BRIGHT_BOARD_COL)
        # draw board
        for row in range(BOARD_HEIGHT):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                y = row * FIELD_SIZE
                x = col * FIELD_SIZE
                pygame.draw.rect(self.window, DARK_BOARD_COL, (x, y, FIELD_SIZE, FIELD_SIZE))


        for row in range(BOARD_HEIGHT):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                cur_col = None
                if self.board.board[row][col].is_white():
                    cur_col = WHITE_PIECES_COL
                elif self.board.board[row][col].is_black():
                    cur_col = BLACK_PIECES_COL
                if cur_col is not None:
                    x = col * FIELD_SIZE
                    y = row * FIELD_SIZE
                    pygame.draw.circle(self.window, cur_col, (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2), PIECE_SIZE)
                    if self.board.board[row][col].is_king():
                        pygame.draw.circle(self.window, KING_MARK_COL, (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2),
                                           PIECE_SIZE / 2)


        if self.something_is_marked:
            x = self.marked_col * FIELD_SIZE
            y = self.marked_row * FIELD_SIZE
            pygame.draw.circle(self.window, POSS_MOVE_MARK_COL, (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2),
                               PIECE_SIZE + MARK_THICK, MARK_THICK)
            pos_moves = self.board.get_piece_moves(self.board.board[self.marked_row][self.marked_col])
            for pos_move in pos_moves:
                self.pos_moves[(pos_move.dest_row, pos_move.dest_col)] = pos_move
                x = pos_move.dest_col * FIELD_SIZE
                y = pos_move.dest_row * FIELD_SIZE
                pygame.draw.circle(self.window, POSS_MOVE_MARK_COL, (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2),
                                   POS_MOVE_MARK_SIZE)

    def update(self):
        self.__draw()
        pygame.display.update()

    def mouse_to_indexes(self, pos):
        return (int(pos[0] // FIELD_SIZE), int(pos[1] // FIELD_SIZE))

    def clicked_at(self, pos):
        (col, row) = self.mouse_to_indexes(pos)
        field = self.board.board[row][col]
        if self.something_is_marked:
            if (row, col) in self.pos_moves:
                self.board.make_move(self.pos_moves[(row, col)])
                self.something_is_marked = False
                self.pos_moves = {}

        if field.is_white():
            if self.something_is_marked:
                self.something_is_marked = False
                self.pos_moves = {}
            else:
                self.something_is_marked = True
                self.marked_col = col
                self.marked_row = row


def main():
    board = Board()
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    is_running = True
    clock = pygame.time.Clock()
    game = Game(window, board)

    while is_running:
        clock.tick(FPS)

        if not game.board.white_turn:
            move = minimax_a_b(game.board, MINIMAX_DEPTH, True, basic_ev_func)
            # move = minimax_a_b( game.board, MINIMAX_DEPTH, True, push_forward_ev_func)
            # move = minimax_a_b( game.board, MINIMAX_DEPTH, True, push_to_opp_half_ev_func)
            # move = minimax_a_b( game.board, MINIMAX_DEPTH, True, group_prize_ev_func)

            if move is not None:
                game.board.make_move(move)
            else:
                is_running = False
        if game.board.end():
            is_running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                game.clicked_at(pos)

        game.update()

    pygame.quit()


def ai_vs_ai(eval, deep1, deep2):
    board = Board()
    is_running = True

    while is_running:
        if board.white_turn:
            move = minimax_a_b(board, deep1, not board.white_turn, basic_ev_func)
        else:
            #move = minimax_a_b(board, 6, not board.white_turn, basic_ev_func)
            move = minimax_a_b( board, deep2, not board.white_turn, eval)
            # move = minimax_a_b( board, 5, not board.white_turn, push_to_opp_half_ev_func)
            # move = minimax_a_b( board, 5, not board.white_turn, group_prize_ev_func)
        if move is not None:
            board.register_move(move)
            board.make_move(move)
        else:
            board.end()
            is_running = False

        if board.end():
            is_running = False
    print(deep1, " ", deep2, " ", eval.__name__)
    print("black_won:", board.black_won)
    print("white_won:", board.white_won)
    # if both won then it is a draw!


#main()
evals = [basic_ev_func, push_forward_ev_func, push_to_opp_half_ev_func, group_prize_ev_func]
deep = [1, 3, 5]

for i in deep:
    for j in deep:
        for k in evals:
            ai_vs_ai(k, i, j)

