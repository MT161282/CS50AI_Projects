import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        self.height = height
        self.width = width
        self.mines = set()

        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        count = 0

        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                if (i, j) == cell:
                    continue

                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game.
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return set(self.cells)

        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return set(self.cells)

        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        self.height = height
        self.width = width

        self.moves_made = set()

        self.mines = set()
        self.safes = set()

        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)

        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)

        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.
        """

        # Record this cell as a move that has been made
        self.moves_made.add(cell)

        # Mark the selected cell as safe
        self.mark_safe(cell)

        neighbors = set()
        adjusted_count = count

        # Check all neighboring cells
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                neighbor = (i, j)

                # Ignore the selected cell itself
                if neighbor == cell:
                    continue

                # Ignore cells outside the board
                if not (0 <= i < self.height and 0 <= j < self.width):
                    continue

                # Adjust count for already known mines
                if neighbor in self.mines:
                    adjusted_count -= 1

                # Add cells that are not already known to be safe
                elif neighbor not in self.safes:
                    neighbors.add(neighbor)

        # Add the new sentence to the knowledge base
        sentence = Sentence(neighbors, adjusted_count)

        if neighbors and sentence not in self.knowledge:
            self.knowledge.append(sentence)

        # Continue making logical deductions
        self._infer_knowledge()

    def _infer_knowledge(self):
        """
        Repeatedly derives mines, safes, and subset sentences
        until no additional information can be inferred.
        """

        changed = True

        while changed:
            changed = False

            # Remove empty and duplicate sentences
            cleaned_knowledge = []

            for sentence in self.knowledge:
                if sentence.cells and sentence not in cleaned_knowledge:
                    cleaned_knowledge.append(sentence)

            if len(cleaned_knowledge) != len(self.knowledge):
                self.knowledge = cleaned_knowledge
                changed = True

            known_safes = set()
            known_mines = set()

            # Find directly known safe cells and mines
            for sentence in self.knowledge:
                known_safes.update(sentence.known_safes())
                known_mines.update(sentence.known_mines())

            new_safes = known_safes - self.safes
            new_mines = known_mines - self.mines

            # Mark newly discovered safe cells
            for safe in new_safes:
                self.mark_safe(safe)
                changed = True

            # Mark newly discovered mines
            for mine in new_mines:
                self.mark_mine(mine)
                changed = True

            if new_safes or new_mines:
                continue

            inferred_sentences = []

            # Infer new sentences using the subset rule
            for first in self.knowledge:
                for second in self.knowledge:

                    if first.cells < second.cells:
                        cells = second.cells - first.cells
                        count = second.count - first.count

                        if not cells:
                            continue

                        if not (0 <= count <= len(cells)):
                            continue

                        inferred = Sentence(cells, count)

                        if (
                            inferred not in self.knowledge
                            and inferred not in inferred_sentences
                        ):
                            inferred_sentences.append(inferred)

            if inferred_sentences:
                self.knowledge.extend(inferred_sentences)
                changed = True

    def make_safe_move(self):
        """
        Returns a safe cell that has not already been chosen.
        """

        available_safes = self.safes - self.moves_made

        if not available_safes:
            return None

        return next(iter(available_safes))

    def make_random_move(self):
        """
        Returns a random move that has not already been chosen
        and is not known to be a mine.
        """

        available_moves = [
            (i, j)
            for i, j in itertools.product(
                range(self.height),
                range(self.width)
            )
            if (i, j) not in self.moves_made
            and (i, j) not in self.mines
        ]

        if not available_moves:
            return None

        return random.choice(available_moves)
