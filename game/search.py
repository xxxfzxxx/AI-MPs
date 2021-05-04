import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    print('moves: ', moves)
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        print('value: ', value)
        print('[move]: ', [move])
        print('{ encode(*move): {} }: ', { encode(*move): {} })
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    ''' 
    moves = [ move for move in generateMoves(side, board, flags) ]
    if depth == 0 or len(moves)==0:
        return (evaluate(board), [], {})
    if not side:
        max_value = -math.inf
        move_tree = {}
        move_list = []
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            value, l, t = minimax(newside, newboard, newflags, depth - 1)
            move_tree[encode(*move)] = t
            if value > max_value:
                max_value = value
                max_move = move
                move_list = l
        move_list.insert(0, max_move)
        
        return (max_value, move_list, move_tree)
    else:
        min_value = math.inf
        move_tree = {}
        move_list = []
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            value, l, t = minimax(newside, newboard, newflags, depth - 1)
            move_tree[encode(*move)] = t
            if value < min_value:
                min_value = value
                min_move = move
                move_list = l
        move_list.insert(0, min_move)
        
        return (min_value, move_list, move_tree)
def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if depth == 0 or len(moves)==0:
        return (evaluate(board), [], {})
    if side == False:
        move_tree = {}
        move_list = []
        target = -math.inf
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            value, l, t = alphabeta(newside, newboard, newflags, depth - 1, alpha, beta)
            move_tree[encode(*move)] = t
            
            if value > target:
                target = value
                alpha = max(target, alpha)
                max_move = move
                move_list = l
                if alpha >= beta:
                    break
        move_list.insert(0, max_move)
        
        return (target, move_list, move_tree)
    else:
        move_tree = {}
        move_list = []
        target = math.inf
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            value, l, t = alphabeta(newside, newboard, newflags, depth - 1, alpha, beta)
            move_tree[encode(*move)] = t
            
            if value < target:
                target = value
                beta = min(target, beta)
                min_move = move
                move_list = l
                if alpha >= beta:
                    break
        move_list.insert(0, min_move)
        
        return (target, move_list, move_tree)

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    
    
    initial_moves = [move for move in generateMoves(side, board, flags)]
    path_avg_score = []
    moveTree = {}
    moveList = []
    moveList_dict = {}
    for m in range(len(initial_moves)):
        newside, newboard, newflags = makeMove(side, board, initial_moves[m][0], initial_moves[m][1], flags, initial_moves[m][2])
        s = newside
        b = newboard
        f = newflags
        breadth_score = []
        curr_list = []
        curr_list.append(initial_moves[m])
        moveTree[encode(*initial_moves[m])] = {}
        for i in range(breadth):
            newside = s
            newflags = f
            newboard = b
            curr_dict = moveTree[encode(*initial_moves[m])]
            for j in range(depth-1):
                random_moves = [move for move in generateMoves(newside, newboard, newflags)]
                random_move = chooser(random_moves)
                if i == 0:
                    curr_list.append(random_move)
                curr_dict[encode(*random_move)] = {}
                curr_dict = curr_dict[encode(*random_move)]
                newside, newboard, newflags = makeMove(newside, newboard, random_move[0], random_move[1], newflags, random_move[2])
            final_score = evaluate(newboard)
            breadth_score.append(final_score) 
        average = sum(breadth_score)/breadth
        path_avg_score.append(average)
        moveList_dict[encode(*initial_moves[m])] = curr_list

    if side:
        best_average = min(path_avg_score)
        best_index = path_avg_score.index(best_average)
    else:
        best_average = max(path_avg_score)
        best_index = path_avg_score.index(best_average)
    
    moveList = []
    best_move = initial_moves[best_index]
    moveList = moveList_dict[encode(*best_move)]
    
    return best_average, moveList, moveTree


