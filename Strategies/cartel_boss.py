"""
The cooperation boss strategy.
"""

cooperation_members = ['MyName', 'Member1', 'Member2']


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
    The cartel will succeed.
    But not as much as me.
    """

    def bid(self, private, public):
        # Default cooperation options
        launch = False
        bid = 1
        # Finds players in the cooperation left
        players, cooperation = [], []
        for i in public['players']:
            if public['players'][i]['bankroll'] > 0:
                players.append(i)
                if i in cooperation_members:
                    cooperation.append(i)
        cooperation.sort()
        if len(players) > len(cooperation):
            # Regular play, every cooperation member takes turns to launch
            if public['round'] % len(cooperation) == cooperation.index(private['name']):
                launch = True
        else:
            # cooperation MODE ACTIVATED
            bid = 0
            if public['auction_round'] == 100:
                launch = True  # All members the same therefore 1/3 of spoils
        # Behind the back changes
        if bid != 0:
            # Regular changes
            if public['base_reward'] < 4 and public['auction_round'] <= 6:
                launch = False
            if public['base_reward'] > 20 or public['round'] == 201:
                launch = True
                bid = 2
            if public['auction_round'] >= 7:
                # Prevents bankruptcy is another cooperation member quits
                bid = 0
        else:
            # cooperation MODE CHANGES - takes 100% of profits if only cartel left.
            if public['auction_round'] == 99:
                launch = True
        # Returns actions to take
        return bid, launch

    def join_launch(self, private, public):
        return False

    def broadcast(self, message):
        print(message)
