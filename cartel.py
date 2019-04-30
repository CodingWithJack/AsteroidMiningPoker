"""
The cartel strategy.
"""

cartel_members = ['MyName', 'Member1', 'Member2']  # The length can be changed


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
    """

    def bid(self, private, public):
        launch = False
        bid = 1
        players, cartel = [], []
        for i in public['players']:
            if public['players'][i]['bankroll'] > 0:
                players.append(i)
                if i in cartel_members:
                    cartel.append(i)
        cartel.sort()
        if len(players) > len(cartel):
            if public['round'] % len(cartel) == cartel.index(private['name']):
                launch = True
        else:
            bid = 0
            if public['auction_round'] == 100:
                launch = True  # All members have same of winning
        return bid, launch

    def join_launch(self, private, public):
        return False

    def broadcast(self, message):
        print(message)
