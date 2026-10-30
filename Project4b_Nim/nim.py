import math
import random
import time


class Nim():

    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize game board.
        Each game board has
            - piles: a list of how many elements remain in each pile
            - player: 0 or 1 to indicate which player's turn
            - winner: None, 0, or 1 to indicate who the winner is
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Nim.available_actions(piles) takes a piles list as input
        and returns all of the available actions (i, j) in that state.

        Action (i, j) represents the action of removing j items
        from pile i (where piles are 0-indexed).
        """
        actions = set()

        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))

        return actions

    @classmethod
    def other_player(cls, player):
        """
        Nim.other_player(player) returns the player that is not
        player. Assumes player is either 0 or 1.
        """
        return 0 if player == 1 else 1

    def switch_player(self):
        """
        Switch the current player to the other player.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Make the move action for the current player.
        action must be a tuple (i, j).
        """
        pile, count = action

        # Check for errors
        if self.winner is not None:
            raise Exception("Game already won")

        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")

        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= count
        self.switch_player()

        # Check for a winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps (state, action)
        pairs to a Q-value (a number).

        - state is a tuple of remaining piles
        - action is a tuple (i, j)
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)

        self.update_q_value(
            old_state,
            action,
            old,
            reward,
            best_future
        )

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state state and action action.
        If no Q-value exists, return 0.
        """
        return self.q.get((tuple(state), action), 0)

    def update_q_value(
        self,
        state,
        action,
        old_q,
        reward,
        future_rewards
    ):
        """
        Update Q-value using the Q-learning formula:

        Q(s, a) = old_q
                  + alpha * (reward + future_rewards - old_q)
        """
        new_value_estimate = reward + future_rewards

        new_q = old_q + self.alpha * (
            new_value_estimate - old_q
        )

        self.q[(tuple(state), action)] = new_q

    def best_future_reward(self, state):
        """
        Return the maximum Q-value among all available actions
        in the given state.

        Return 0 if there are no available actions.
        """
        actions = Nim.available_actions(state)

        if not actions:
            return 0

        return max(
            self.get_q_value(state, action)
            for action in actions
        )

    def choose_action(self, state, epsilon=True):
        """
        Choose an available action.

        If epsilon is False, choose the action with the highest
        Q-value.

        If epsilon is True, choose a random action with probability
        self.epsilon. Otherwise choose the action with the highest
        Q-value.
        """
        actions = Nim.available_actions(state)

        if not actions:
            return None

        # Exploration: choose a random action
        if epsilon and random.random() < self.epsilon:
            return random.choice(tuple(actions))

        # Exploitation: choose the action with the highest Q-value
        return max(
            actions,
            key=lambda action: self.get_q_value(state, action)
        )


def train(n):
    """
    Train an AI by playing n games against itself.
    """
    player = NimAI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # Keep track of last move made by either player
        last = {
            0: {
                "state": None,
                "action": None
            },
            1: {
                "state": None,
                "action": None
            }
        }

        # Game loop
        while True:

            # Keep track of current state and action
            state = game.piles.copy()
            action = player.choose_action(game.piles)

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move
            game.move(action)
            new_state = game.piles.copy()

            # When game is over, update Q-values with rewards
            if game.winner is not None:

                # The player who took the last object loses
                player.update(
                    state,
                    action,
                    new_state,
                    -1
                )

                # The other player wins
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )

                break

            # If game continues, update previous move with reward 0
            elif last[game.player]["state"] is not None:

                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")

    return player


def play(ai, human_player=None):
    """
    Play a human game against the AI.

    human_player can be 0 or 1 to specify whether
    the human player moves first or second.
    """

    # Choose human player order randomly if not specified
    if human_player is None:
        human_player = random.randint(0, 1)

    game = Nim()

    # Game loop
    while True:

        # Print piles
        print()
        print("Piles:")

        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")

        print()

        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        # Human's turn
        if game.player == human_player:
            print("Your Turn")

            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))

                if (pile, count) in available_actions:
                    break

                print("Invalid move, try again.")

        # AI's turn
        else:
            print("AI's Turn")

            pile, count = ai.choose_action(
                game.piles,
                epsilon=False
            )

            print(
                f"AI chose to take {count} from pile {pile}."
            )

        # Make move
        game.move((pile, count))

        # Check winner
        if game.winner is not None:
            print()
            print("GAME OVER")

            winner = (
                "Human"
                if game.winner == human_player
                else "AI"
            )

            print(f"Winner is {winner}")
            return

