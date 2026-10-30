import sys
from collections import deque

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generator.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]

        for variable, word in assignment.items():
            direction = variable.direction

            for k in range(len(word)):
                i = variable.i + (
                    k if direction == Variable.DOWN else 0
                )
                j = variable.j + (
                    k if direction == Variable.ACROSS else 0
                )
                letters[i][j] = word[k]

        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")

            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont

        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        img = Image.new(
            "RGBA",
            (
                self.crossword.width * cell_size,
                self.crossword.height * cell_size
            ),
            "black"
        )

        font = ImageFont.truetype(
            "assets/fonts/OpenSans-Regular.ttf",
            80
        )
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (
                        j * cell_size + cell_border,
                        i * cell_size + cell_border
                    ),
                    (
                        (j + 1) * cell_size - cell_border,
                        (i + 1) * cell_size - cell_border
                    )
                ]

                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")

                    if letters[i][j]:
                        _, _, w, h = draw.textbbox(
                            (0, 0),
                            letters[i][j],
                            font=font
                        )

                        draw.text(
                            (
                                rect[0][0]
                                + ((interior_size - w) / 2),
                                rect[0][1]
                                + ((interior_size - h) / 2)
                                - 10
                            ),
                            letters[i][j],
                            fill="black",
                            font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, then solve the CSP.
        """
        self.enforce_node_consistency()

        if not self.ac3():
            return None

        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Remove words that do not match a variable's length.
        """
        for variable in self.crossword.variables:
            self.domains[variable] = {
                word
                for word in self.domains[variable]
                if len(word) == variable.length
            }

    def revise(self, x, y):
        """
        Make x arc-consistent with y.

        Return True if x's domain was changed.
        Otherwise, return False.
        """
        overlap = self.crossword.overlaps[x, y]

        if overlap is None:
            return False

        x_index, y_index = overlap
        revised = False

        for x_word in self.domains[x].copy():
            has_match = any(
                x_word[x_index] == y_word[y_index]
                for y_word in self.domains[y]
            )

            if not has_match:
                self.domains[x].remove(x_word)
                revised = True

        return revised

    def ac3(self, arcs=None):
        """
        Enforce arc consistency using AC-3.

        Return False if a domain becomes empty.
        Otherwise, return True.
        """
        if arcs is None:
            queue = deque(
                (x, y)
                for x in self.crossword.variables
                for y in self.crossword.neighbors(x)
            )
        else:
            queue = deque(arcs)

        while queue:
            x, y = queue.popleft()

            if self.revise(x, y):

                if not self.domains[x]:
                    return False

                for neighbor in (
                    self.crossword.neighbors(x) - {y}
                ):
                    queue.append((neighbor, x))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if every variable has been assigned.
        """
        return all(
            variable in assignment
            for variable in self.crossword.variables
        )

    def consistent(self, assignment):
        """
        Return True if assignment satisfies all constraints.
        """
        words = list(assignment.values())

        # Every word must be different
        if len(words) != len(set(words)):
            return False

        for variable, word in assignment.items():

            # Word must have the correct length
            if len(word) != variable.length:
                return False

            # Check every assigned neighboring variable
            for neighbor in self.crossword.neighbors(variable):

                if neighbor not in assignment:
                    continue

                variable_index, neighbor_index = (
                    self.crossword.overlaps[
                        variable,
                        neighbor
                    ]
                )

                if (
                    word[variable_index]
                    != assignment[neighbor][neighbor_index]
                ):
                    return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return values ordered by the
        least-constraining-value heuristic.
        """
        eliminated_counts = {}

        for value in self.domains[var]:
            eliminated = 0

            for neighbor in self.crossword.neighbors(var):

                if neighbor in assignment:
                    continue

                var_index, neighbor_index = (
                    self.crossword.overlaps[
                        var,
                        neighbor
                    ]
                )

                for neighbor_value in self.domains[neighbor]:

                    if (
                        value[var_index]
                        != neighbor_value[neighbor_index]
                    ):
                        eliminated += 1

            eliminated_counts[value] = eliminated

        return sorted(
            self.domains[var],
            key=lambda value: (
                eliminated_counts[value],
                value
            )
        )

    def select_unassigned_variable(self, assignment):
        """
        Select a variable using MRV and then degree heuristic.
        """
        unassigned = [
            variable
            for variable in self.crossword.variables
            if variable not in assignment
        ]

        return min(
            unassigned,
            key=lambda variable: (
                len(self.domains[variable]),
                -len(
                    self.crossword.neighbors(variable)
                )
            )
        )

    def backtrack(self, assignment):
        """
        Use backtracking search to find a complete assignment.
        """
        if self.assignment_complete(assignment):
            return assignment

        variable = self.select_unassigned_variable(
            assignment
        )

        for value in self.order_domain_values(
            variable,
            assignment
        ):
            assignment[variable] = value

            if self.consistent(assignment):
                result = self.backtrack(assignment)

                if result is not None:
                    return result

            assignment.pop(variable)

        return None


def main():

    if len(sys.argv) not in [3, 4]:
        sys.exit(
            "Usage: python generate.py structure words [output]"
        )

    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    if assignment is None:
        print("No solution.")

    else:
        creator.print(assignment)

        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
