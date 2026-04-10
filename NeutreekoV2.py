
# importations
from copy import deepcopy
from random import choice
from time import sleep
import matplotlib.pyplot as mp
import numpy as np


class Grid:
    def __init__(self):
        mp.axis('equal')  # square grid
        ax.clear()
        for i in range(6):
            jx, jy = [0, 5], [i, i]
            mp.plot(jx, jy, "k")  # columns
            ix, iy = [i, i], [0, 5]
            mp.plot(ix, iy, "k")  # lines
        mp.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False,
                       labelleft=False)  # get rid of ticks
        mp.title('Neutreeko', fontsize=25, family='cursive', backgroundcolor='white')
        mp.text(-1.3, 5.8, 'Your turn:', horizontalalignment='center', verticalalignment='center', fontsize=10,
                weight='bold')
        mp.show()


class Pawn:
    def __init__(self, pawn_number, colour):
        self.pawn_number = pawn_number
        self.colour = colour
        self.line, self.column = self.getPosition(matrix)  # i+1 car n>=1
        self.can_move = [self.canMoveRight(), self.canMoveLeft(), self.canMoveUp(),
                         self.canMoveDown(), self.canMoveUpLeft(), self.canMoveUpRight(),
                         self.canMoveDownLeft(), self.canMoveDownRight()]
        self.pawn = mp.Rectangle((self.column, self.line), 1, 1, 0, facecolor=self.colour, edgecolor="k",
                                 fill=True)  # ((bottom left summit), width, height, angle, colour, edge colour, full)
        ax.add_patch(self.pawn)  # add pawn in grid
        mp.show()

    def getPosition(self, p_matrix):
        for i in range(5):
            for j in range(5):
                if p_matrix[i, j] == self.pawn_number:
                    self.line = i
                    self.column = j
        return self.line, self.column

    def canMoveRight(self):
        nb_of_moves = 0  # number of squares to move
        while self.column + nb_of_moves < 4 and matrix[self.line, self.column + nb_of_moves + 1] == 0:  # while can move
            nb_of_moves += 1
        return self.line, self.column + nb_of_moves

    def canMoveLeft(self):
        nb_of_moves = 0  # number of squares to move
        while self.column - nb_of_moves > 0 and matrix[self.line, self.column - nb_of_moves - 1] == 0:  # while can move
            nb_of_moves += 1
        return self.line, self.column - nb_of_moves

    def canMoveDown(self):
        nb_of_moves = 0  # number of squares to move
        while self.line - nb_of_moves > 0 and matrix[self.line - nb_of_moves - 1, self.column] == 0:  # while can move
            nb_of_moves += 1
        return self.line - nb_of_moves, self.column

    def canMoveUp(self):
        nb_of_moves = 0  # number of squares to move
        while self.line + nb_of_moves < 4 and matrix[self.line + nb_of_moves + 1, self.column] == 0:  # while can move
            nb_of_moves += 1
        return self.line + nb_of_moves, self.column

    def canMoveDownLeft(self):
        nb_of_moves = 0  # number of squares to move
        while self.column - nb_of_moves > 0 and self.line - nb_of_moves > 0 \
                and matrix[self.line - nb_of_moves - 1, self.column - nb_of_moves - 1] == 0:  # while can move
            nb_of_moves += 1
        return self.line - nb_of_moves, self.column - nb_of_moves

    def canMoveDownRight(self):
        nb_of_moves = 0  # number of squares to move
        while self.column + nb_of_moves < 4 and self.line - nb_of_moves > 0 \
                and matrix[self.line - nb_of_moves - 1, self.column + nb_of_moves + 1] == 0:  # while can move
            nb_of_moves += 1
        return self.line - nb_of_moves, self.column + nb_of_moves

    def canMoveUpLeft(self):
        nb_of_moves = 0  # number of squares to move
        while self.column - nb_of_moves > 0 and self.line + nb_of_moves < 4 \
                and matrix[self.line + nb_of_moves + 1, self.column - nb_of_moves - 1] == 0:  # while can move
            nb_of_moves += 1
        return self.line + nb_of_moves, self.column - nb_of_moves

    def canMoveUpRight(self):
        nb_of_moves = 0  # number of squares to move
        while self.column + nb_of_moves < 4 and self.line + nb_of_moves < 4 \
                and matrix[self.line + nb_of_moves + 1, self.column + nb_of_moves + 1] == 0:  # while can move
            nb_of_moves += 1
        return self.line + nb_of_moves, self.column + nb_of_moves


def move(pawn, i, p_matrix):
    index = directions.index(i)
    new_line, new_column = pawn.can_move[index]
    p_matrix[pawn.line, pawn.column] = 0
    pawn.line, pawn.column = new_line, new_column
    p_matrix[pawn.line, pawn.column] = pawn.pawn_number
    if not np.array_equal(p_matrix, matrix):
        # noinspection PyBroadException
        try:
            print("it")
            pawn.pawn.remove()
        except Exception:
            pass
    setup()


def hasWon(p_matrix, i, j, k):
    """Detects if a player has won"""
    (a, z) = i.getPosition(p_matrix)
    (e, r) = j.getPosition(p_matrix)
    (t, q) = k.getPosition(p_matrix)
    if a == e and a == t:  # 3!=6 horizontal possibilities
        if (z == r+1 and z == q+2) or (z == r-1 and z == q-2) or (z == r-1 and z == q+1) \
                or (z == r+1 and z == q-1) or (z == q+1 and z == r+2) or (z == q-1 and z == r-2):
            return True
    elif z == r and z == q:  # 3!=6 vertical possibilities
        if (a == e+1 and a == t+2) or (a == e-1 and a == t-2) or (a == e-1 and a == t+1) \
                or (a == e+1 and a == t-1) or (a == t+1 and a == e+2) or (a == t-1 and a == e-2):
            return True
    else:  # 2*3!=12 diagonal possibilities
        if ((a, z) == (e+1, r+1) and (a, z) == (t+2, q+2)) or ((a, z) == (e+2, r+2) and (a, z) == (t+1, q+1)) \
                or ((a, z) == (e-1, r-1) and (a, z) == (t-2, q-2)) or ((a, z) == (e-2, r-2) and (a, z) == (t-1, q-1)) \
                or ((a, z) == (e-1, r-1) and (a, z) == (t+1, q+1)) or ((a, z) == (e+1, r+1) and (a, z) == (t-1, q-1)) \
                or ((a, z) == (e-1, r+1) and (a, z) == (t-2, q+2)) or ((a, z) == (e+1, r-1) and (a, z) == (t+2, q-2)) \
                or ((a, z) == (e-2, r+2) and (a, z) == (t-1, q+1)) or ((a, z) == (e+2, r-2) and (a, z) == (t+1, q-1)) \
                or ((a, z) == (e-1, r+1) and (a, z) == (t+1, q-1)) or ((a, z) == (e+1, r-1) and (a, z) == (t-1, q+1)):
            return True
    return False


def selection(event):
    """Detects pawn to move"""
    global x, y
    y, x = int(event.xdata), int(event.ydata)


def commands(event):
    """Switches player turn"""
    global iterator
    pawn_number = matrix[x, y]
    pawn = pawns[pawn_number - 1]
    direction = event.key
    print("hello")
    if iterator % 2 == 0 and pawn_number in (4, 5, 6):  # blue's turn
        matrix_copy = deepcopy(matrix)
        move(pawn, direction, matrix)
        if np.array_equal(matrix_copy, matrix):
            commands(event)
        setup()
        if hasWon(matrix, p4, p5, p6):
            print("victory")
            mp.ioff()
        else:
            mp.pause(0.2)
            iterator += 1
            sleep(1)
            commands(event)
    elif iterator % 2 != 0:  # red's turn
        if difficulty == 1:
            level1()
            if hasWon(matrix, p1, p2, p3):
                print("victory")
                mp.ioff()
            else:
                setup()
                iterator += 1
                commands(event)
        if difficulty == 2:
            level2()
            print("hello")
            if hasWon(matrix, p1, p2, p3):
                print("victory")
                mp.ioff()
            else:
                setup()
                sleep(1)
                iterator += 1
                commands(event)
        if difficulty == 3:
            level3()
            if hasWon(matrix, p1, p2, p3):
                print("victory")
                mp.ioff()
            else:
                setup()
                iterator += 1
                commands(event)
        if difficulty == 4 and pawn_number in (1, 2, 3):
            move(pawn, direction, matrix)
            setup()
            if hasWon(matrix, p1, p2, p3):
                print("victory")
                mp.ioff()
            else:
                iterator += 1
                commands(event)


# Artificial Intelligence

def randomMove():
    """Makes random move"""
    pawn = choice([p1, p2, p3])
    direction = choice(directions)
    v_matrix = deepcopy(matrix)
    line, column = pawn.getPosition(matrix)
    move(pawn, direction, matrix)
    # if not np.array_equal(v_matrix, matrix):
    #    matrix[line, column] = 0
    line2, column2 = pawn.getPosition(matrix)
    if (line, column) == (line2, column2):
        randomMove()
    setup()


def check4(v_matrix, pawn_number):
    """Checks that that the move puts pawns side by side"""
    print("check4")
    for i in (p1, p2, p3):
        if i == pawns[pawn_number - 1]:  # don't compare a pawn with itself
            continue
        v_matrix2 = deepcopy(v_matrix)
        line1, column1 = i.getPosition(v_matrix2)
        line2, column2 = pawns[pawn_number - 1].getPosition(v_matrix2)
        if (line2 == line1+1 and column2 == column1) or (line2 == line1-1 and column2 == column1) \
                or (line2 == line1 and column2 == column1+1) or (line2 == line1 and column2 == column1-1) \
                or (line2 == line1+1 and column2 == column1+1) or (line2 == line1-1 and column2 == column1+1) \
                or (line2 == line1+1 and column2 == column1-1) or (line2 == line1-1 and column2 == column1-1):
            return True
    return False


def check3(v_matrix, pawn_number):
    """Detects if by making a move, the player can win."""
    # print("check3")
    # print("boo", v_matrix)
    # for direction in directions:
    #     v_matrix2 = deepcopy(v_matrix)
    #     print("check3", v_matrix2)
    #
    #
    #     line, column = pawns[pawn_number - 1].getPosition(v_matrix)
    #     pawns[pawn_number - 1].move(direction, v_matrix2)
    #     if not np.array_equal(v_matrix, v_matrix2):
    #         v_matrix[line, column] = 0
    #     else:
    #         continue
    #     print(v_matrix2, "check3")
    line, column = pawns[pawn_number - 1].getPosition(v_matrix)
    print(pawn_number)
    if checkV2(v_matrix, line, column, pawns[pawn_number - 1], 3, False):
        print("True")
        return True
    return False


def verif3(v_matrix, pawn_number):
    """détecte si en faisant un déplacement le joueur(vert) peut gagner."""
    pawn = pawns[pawn_number - 1]
    for direction in directions:  # parcourt les sens de déplacement possibles
        v_matrix2 = deepcopy(v_matrix)
        print(v_matrix2)
        move(pawn, direction, v_matrix2)
        print(v_matrix2, pawn_number, direction, "verif3")
        if hasWon(v_matrix2, p4, p5, p6):
            return True
    return False

def check2(v_matrix):
    """Checks that player can win. If so, return true."""
    print("check2")
    for pawn_number in (4, 5, 6):
        if verif3(v_matrix, pawn_number):
            print("True2")
            return True
    return False


def checkV2(p_matrix, line, column, pawn, player_number, move=True):
    """Check if can win in one move. move=True to make the move."""
    print("check", pawn)
    for possible_move in pawn.can_move:  # possible moves
        print("check", matrix)
        index = pawn.can_move.index(possible_move)
        print("check1", matrix)
        direction = directions[index]
        v_matrix = deepcopy(p_matrix)
        line2, column2 = pawn.can_move[index]
        if (line, column) != (line2, column2):  # if the move is possible
            print(v_matrix, "check")
            v_matrix = pawn.move(direction, v_matrix)
            print(v_matrix, "check1")
            v_matrix[line, column] = 0
            piece1, piece2, piece3 = p1, p2, p3
            if player_number == 3:
                piece1, piece2, piece3 = p4, p5, p6
            if hasWon(v_matrix, piece1, piece2, piece3):  # if th AI can win
                print("ia can win")
                if move:  # if move=True, make the move
                    print("move")
                    print(matrix, "version 1")
                    pawn.move(direction, matrix)
                    print("version 1.0", matrix)
                    matrix[line, column] = 0
                    print(matrix, "version 1.0")
                    setup()
                    mp.pause(0.1)
                return True
    print("False")
    return False


def check(line, column, pawn):
    for possible_move in pawn.can_move:  # possible moves
        index = pawn.can_move.index(possible_move)
        direction = directions[index]
        v_matrix = deepcopy(matrix)
        line2, column2 = pawn.can_move[index]
        if (line, column) != (line2, column2):  # if the move is possible
            move(pawn, direction, v_matrix)
            print(v_matrix)
            v_matrix[line, column] = 0
            print(v_matrix, "boo", pawn.pawn_number, direction)
            if hasWon(v_matrix, p1, p2, p3):  # if th AI can win
                move(pawn, direction, matrix)
                matrix[line, column] = 0
                setup()
                mp.pause(0.1)
                print("true dat")
                return True
    print("False")
    return False


def canWinInOneMove():  # player 1 = ia, player 3 is person
    i = 1
    print("can")
    while i < 4:
        line, column = pawns[i - 1].getPosition(matrix)
        if check(line, column, pawns[i - 1]):
            print("can win")
            return True
        i += 1


def level1():
    """Detects if AI can win in one move. If so, make the move."""
    if canWinInOneMove():
        print("level1")
        return
    randomMove()
    setup()
    mp.pause(0.2)
    return


def level2():
    """Stops player from winning. If not possible, make random move."""
    global limit, iterator
    print(limit, "level2")
    if canWinInOneMove():
        limit = 0
        return None
    limit += 1
    while limit < 400:
        print("while")
        pawn_number = choice([1, 2, 3])
        direction = choice(directions)
        v_matrix = deepcopy(matrix)
        line, column = pawns[pawn_number - 1].getPosition(matrix)
        print(matrix, "matrix2")
        move(pawns[pawn_number - 1], direction, v_matrix)
        print(matrix, "matrix", pawn_number, direction)

        if not np.array_equal(v_matrix, matrix):
            v_matrix[line, column] = 0
        line2, column2 = pawns[pawn_number - 1].getPosition(v_matrix)
        if np.array_equal(v_matrix, matrix):  # if move wasn't possible
            print("impossible move")
            level2()
        elif not check2(v_matrix):      # check(v_matrix, line2, column2, pawns[pawn_number - 1], 3, move=False):
            print("losing move")
            level2()
        else:  # if move was possible and doesn't let player win, make the move
            print("block")
            limit = 0
            print(matrix, "else", pawn_number, direction)
            move(pawns[pawn_number - 1], direction, matrix)
            matrix[line, column] = 0
            print("else", matrix)
            setup()
            mp.show()
            return None
    print("cannot block")
    limit = 0
    randomMove()  # all possibilities have been tested, player will win
    return None


def level3():
    """Stops player from winning and tries to put pawns side by side. If not possible, make random move."""
    global limit
    limit += 1
    while limit < 400:
        v_matrix = deepcopy(matrix)
        pawn_number = choice([1, 2, 3])
        direction = choice(directions)
        v_matrix = pawns[pawn_number - 1].move(direction, v_matrix)
        if np.array_equal(v_matrix, matrix):  # if move wasn't possible
            level3()
        elif check2(v_matrix):
            level3()
        else:  # if move was possible and doesn't let player win, make the move
            if check4(v_matrix, pawn_number):
                limit = 0
                pawns[pawn_number - 1].move(direction, matrix)
                return None
            else:
                level3()
    limit = 0
    level2()
    return None


def play():
    """Connects figure to commands and selection to make interactive"""
    print("play")
    setup()
    fig.canvas.mpl_connect('button_release_event', selection)
    fig.canvas.mpl_connect('key_press_event', commands)


def setup():
    global grid, pawns, p1, p2, p3, p4, p5, p6
    grid = Grid()
    p1 = Pawn(1, 'red')
    p2 = Pawn(2, 'red')
    p3 = Pawn(3, 'red')
    p4 = Pawn(4, 'blue')
    p5 = Pawn(5, 'blue')
    p6 = Pawn(6, 'blue')
    pawns = [p1, p2, p3, p4, p5, p6]


def levelSelection(prompt):
    """Checks that entered value is correct"""
    while True:
        try:
            num = float(input(prompt))
            break
        except ValueError:
            pass
    return num


def displayMenu():
    """Displays level options in console"""
    global difficulty
    for i in range(len(options)):
        print("{:d}. {:s}".format(i + 1, options[i]))
    select = 0
    while select not in (1, 2, 3, 4):
        select = levelSelection("Choose difficulty: ")
    difficulty = select
    play()


# Main script

# Attributes
global grid, p1, p2, p3, p4, p5, p6, pawns, x, y, difficulty
fig = mp.figure()
mp.ion()
ax = mp.axes()
matrix = np.array([[0, 1, 0, 2, 0], [0, 0, 4, 0, 0], [0, 0, 0, 0, 0], [0, 0, 3, 0, 0], [0, 5, 0, 6, 0]])
directions = ['right', 'left', 'up', 'down', 'a', 'e', 'w', 'c']
options = np.array(["Easy", "Normal", "Hard", "PVP"])

# Play

iterator = 0
limit = 0
displayMenu()
mp.show(block=True)
