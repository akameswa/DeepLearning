import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        last_symbol = None

        for i in range(len(y_probs[0])):
            max_prob = 0
            for j in range(len(y_probs)):
                prob = y_probs[j][i]
                if prob > max_prob:
                    max_prob = prob
                    max_symbol = j

            path_prob *= max_prob

            if max_symbol != blank and (last_symbol != self.symbol_set[max_symbol - 1]):
                decoded_path.append(self.symbol_set[max_symbol - 1])
                
            last_symbol = self.symbol_set[max_symbol - 1]

        decoded_path = ''.join(decoded_path)
        return decoded_path, path_prob

class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        T = y_probs.shape[1]

        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = self.InitializePaths(self.symbol_set, y_probs[:, 0, :])

        for t in range(1, T):
            PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = self.Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, self.beam_width)
            NewPathsWithTerminalBlank, NewBlankPathScore = self.ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:, t, :], BlankPathScore, PathScore)
            NewPathsWithTerminalSymbol, NewPathScore = self.ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, self.symbol_set, y_probs[:, t, :], BlankPathScore, PathScore)

        _, FinalPathScore = self.MergeIdenticalPaths(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore)

        bestPath = max(FinalPathScore, key=lambda path: FinalPathScore[path])
        return bestPath, FinalPathScore


    def InitializePaths(self, SymbolSet, y):
        InitialBlankPathScore = {}
        InitialPathScore = {}

        path = ""
        InitialBlankPathScore[path] = y[0]
        InitialPathsWithFinalBlank = []
        InitialPathsWithFinalBlank.append(path)

        InitialPathWithFinalSymbol = []

        for i, c in enumerate(SymbolSet):
            path = c
            InitialPathScore[path] = y[i + 1]
            InitialPathWithFinalSymbol.append(path)

        return InitialPathsWithFinalBlank, InitialPathWithFinalSymbol, InitialBlankPathScore, InitialPathScore


    def ExtendWithBlank(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore):
        UpdatedPathsWithTerminalBlank = []
        UpdatedBlankPathScore = {}

        for path in PathsWithTerminalBlank:
            UpdatedPathsWithTerminalBlank.append(path)
            UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]

        for path in PathsWithTerminalSymbol:
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += PathScore[path] * y[0]
            else:
                UpdatedPathsWithTerminalBlank.append(path)
                UpdatedBlankPathScore[path] = PathScore[path] * y[0]

        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

    def ExtendWithSymbol(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y, BlankPathScore, PathScore):
        UpdatedPathsWithTerminalSymbol = set()
        UpdatedPathScore = {}

        for path in PathsWithTerminalBlank:
            for i, c in enumerate(SymbolSet):
                newPath = path + c
                UpdatedPathsWithTerminalSymbol.add(newPath)
                UpdatedPathScore[newPath] = BlankPathScore[path] * y[i + 1]

        for path in PathsWithTerminalSymbol:
            for i, c in enumerate(SymbolSet):
                newPath = path if (c == path[-1]) else path + c
                if newPath in UpdatedPathsWithTerminalSymbol:
                    UpdatedPathScore[newPath] += PathScore[path] * y[i + 1]
                else:
                    UpdatedPathsWithTerminalSymbol.add(newPath)
                    UpdatedPathScore[newPath] = PathScore[path] * y[i + 1]

        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
        PrunedBlankPathScore = {}
        PrunedPathScore = {}
        
        scorelist = []
        for p in PathsWithTerminalBlank:
            scorelist.append(BlankPathScore[p])
        for p in PathsWithTerminalSymbol:
            scorelist.append(PathScore[p])

        scorelist.sort(reverse=True)
        cutoff = scorelist[BeamWidth] if (BeamWidth < len(scorelist)) else scorelist[-1]

        PrunedPathsWithTerminalBlank = []
        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] > cutoff:
                PrunedPathsWithTerminalBlank.append(p)
                PrunedBlankPathScore[p] = BlankPathScore[p]
                            
        PrunedPathsWithTerminalSymbol = []
        for p in PathsWithTerminalSymbol:
            if PathScore[p] > cutoff:
                PrunedPathsWithTerminalSymbol.append(p)
                PrunedPathScore[p] = PathScore[p]

        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

    def MergeIdenticalPaths(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore

        for p in PathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] += BlankPathScore[p]
            else:
                MergedPaths.add(p)
                FinalPathScore[p] = BlankPathScore[p]
                
        return MergedPaths, FinalPathScore