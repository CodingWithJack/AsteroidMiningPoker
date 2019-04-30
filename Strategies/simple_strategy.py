"""
strategies.py

Put all strategies in here and import into main file.

A strategy needs to implement .bid() and .join_launch() methods.

Optional methods are .begin() and .end() called at the start and
end of a game (or bankruptcy of the player), respectively, as
well as .broadcast(), which receives human readable messages
about the game's progress.
"""


class Strategy(object):
    """
    Template strategy, which specific strategies inherit.
    """
    def bid(self, private_information, public_information):
        raise Exception("you need to implement a bid strategy!")

    def join_launch(self, private_information, public_information):
        raise Exception("you need to implement a launch strategy!")

    def update_status(self, private_information, public_information):
        # assume bot, no reaction to game state updates
        return False

    def broadcast(self, message):
        # assume bot, so no messages necessary
        return False

    def ping(self):
        return True


class StudentStrategy(Strategy):
    """
    My strategy
    """
    def bid(self, private_information, public_information):
        tech = private_information["tech"]
        base = public_information["base_reward"]
        if (tech > 4 and base > 5) or tech > 10:
            return 5, True
        else:
            return 4, False

    def join_launch(self, private_information, public_information):
        return False

    def broadcast(self, message):
        print(message)
