"""
strategies.py

Put all strategies in here and import into main file.

A strategy needs to implement .bid() and .join_launch() methods.
"""

import numpy as np
import json
import datetime as dt
import scipy.integrate as integrate

# Following variables change depending on running server
server = 'hearst/'
turns = 300


class PlayerModel:

    all_players = []

    def initialize(player):
        # Makes sure all players are accounted for
        if player not in PlayerModel.all_players:
            PlayerModel.all_players.append(player)
        try:
            with open(server+player+'.json', 'r') as json_save:
                save = json.load(json_save)
                json_save.close()
            # Reset tech, last launch round, and reason
            save['tech'] = 0
            save['last_launch_round'] = -10000
            save['reason'] = 0
            # Reset some values (132 games analysed a day with 1/10 chance)
            # Makes sure optimization doesn't negatively effect major log
            time = dt.datetime.now().hour
            if time < 23:
                reset = np.random.randint(100)
                if reset <= 1 and save['start_round'] < turns:
                    save['start_round'] += 1  # 2 times in 100
                elif reset <= 3 and save['min_launch_gap'] < turns:
                    save['min_launch_gap'] += 1  # 2 times in 100
                elif reset <= 5 and save['max_launch_gap'] > 0:
                    save['max_launch_gap'] -= 1  # 2 times in 100
                elif reset <= 7 and save['max_tech'] > 0:
                    save['max_tech'] -= 1  # 2 times in 100
                elif reset <= 8:
                    # 1 time in 100
                    always_launch_round = np.random.randint(turns) + 1
                    if always_launch_round not in save['always_launch']:
                        save['always_launch'].append(always_launch_round)
                elif reset <= 10 and save['min_launch_price'] > 0:
                    save['min_launch_price'] += 1  # 2 times in 100
                elif reset <= 12 and save['max_nl_price'] > 0:
                    save['max_nl_price'] -= 1  # 2 times in 100
                elif reset == 13 and time == 2:
                    # resets optimizer completely 6% chance a day
                    save['epsilon_s'] = list(np.linspace(0, 0, 21))
                    save['epsilon_f'] = list(np.linspace(0, 0, 21))
        except FileNotFoundError:
            """
            1: lowest turn a stategy possibly launches
            2: minimum number of turns inbetween a strategies launches
            3: maximum number of turns between a strategies launches
            4: maximum tech value ever had before launching
            5: rounds on which a strategy will always launch
            6: maximum base value for which a strategy will never launch
            7: minimum base value for which a strategy won't always launch
            8: guess whether a strategy will launch based on their tech ratio
            """
            save = {'start_round': turns,  # 1
                    'min_launch_gap': turns,  # 2
                    'max_launch_gap': 0,  # 3
                    'max_tech': 0,  # 4
                    'always_launch': list(np.linspace(1, turns, turns)),  # 5
                    'min_launch_price': 10000,  # new 6
                    'max_nl_price': 0,  # new 7
                    'epsilon': -10000,  # 8-9
                    'epsilon_s': list(np.linspace(0, 0, 21)),
                    'epsilon_f': list(np.linspace(0, 0, 21)),  # Range of 21.
                    'tech': 0,
                    'last_launch_round': -10000,
                    'reason': 0}  # Reason for last launch (0 none) ^
        with open(server+player+'.json', 'w') as json_save:
            json.dump(save, json_save)
            json_save.close()
        return None

    def set_tech(players, tech, plus=True):
        # Does not model my own tech
        players = list(set(players) & set(PlayerModel.all_players))
        # Adds tech for all players who won
        for i in players:
            with open(server+i+'.json', 'r') as json_save:
                save = json.load(json_save)
                json_save.close()
            if plus:
                save['tech'] += tech
            else:
                save['tech'] = tech
            with open(server+i+'.json', 'w') as json_save:
                json.dump(save, json_save)
                json_save.close()
        return None

    def business_done():
        PlayerModel.set_tech(PlayerModel.all_players, 5)
        return None

    def auction_done(players):
        PlayerModel.set_tech(players, 5)
        return None

    def launch_done(players, public):
        for i in PlayerModel.all_players:
            with open(server+i+'.json', 'r') as json_save:
                save = json.load(json_save)
                json_save.close()
            launch_gap = public['round'] - save['last_launch_round']
            if i in players:
                # Set last launch round to current round
                save['last_launch_round'] = public['round']
                # Check for optimize mistakes
                if save['reason'] == 1:
                    save['start_round'] = public['round']
                elif save['reason'] == 2:
                    save['min_launch_gap'] = launch_gap
                elif save['reason'] == 6:
                    save['min_launch_price'] = public['base_reward']
                elif save['reason'] == 8:
                    save['epsilon_s'][save['epsilon']] += 1
                    # Sucessful guess
                elif save['reason'] == 9:
                    save['epsilon_f'][save['epsilon']] += 1
                    # Failed guess
            if i not in players and save['reason'] != 0:
                if save['reason'] == 3:
                    save['max_launch_gap'] = launch_gap
                elif save['reason'] == 4:
                    save['max_tech'] = save['tech']
                elif save['reason'] == 5:
                    save['always_launch'].remove(public['round'])
                elif save['reason'] == 7:
                    save['max_nl_price'] = public['base_reward']
                elif save['reason'] == 8:
                    save['epsilon_f'][save['epsilon']] += 1
                    # Failed guess
                elif save['reason'] == 9:
                    save['epsilon_s'][save['epsilon']] += 1
                    # Sucessful guess
            save['reason'] = 0
            with open(server+i+'.json', 'w') as json_save:
                json.dump(save, json_save)
                json_save.close()
        return None

    def mining_done(players):
        PlayerModel.set_tech(players, 0, plus=False)
        return None

    def launching(player, public, p_tech):
        # Guesses if a player is going to launch
        launch = False
        with open(server+player+'.json', 'r') as json_save:
            save = json.load(json_save)
            json_save.close()
        # Finds a reason the player will launch
        reason = 0
        llr = save['last_launch_round']
        if public['round'] < save['start_round']:
            reason = 1
        elif llr >= 0 and (public['round'] - llr) < save['min_launch_gap']:
            reason = 2
        elif llr >= 0 and (public['round'] - llr) > save['max_launch_gap']:
            launch = True
            reason = 3
        elif p_tech > save['max_tech']:
            launch = True
            reason = 4
        elif public['round'] in save['always_launch']:
            launch = True
            reason = 5
        elif public['base_reward'] < save['min_launch_price']:
            launch = False
            reason = 6
        elif public['base_reward'] > save['max_nl_price']:
            launch = True
            reason = 7
        else:
            maximise = []
            e_s = save['epsilon_s']
            for i in range(len(e_s)):
                if e_s[i] == 0:
                    maximise.append(0)
                else:
                    maximise.append(e_s[i] / (e_s[i] + save['epsilon_f'][i]))
            inds = [i for i, x in enumerate(maximise) if x == max(maximise)]
            time = dt.datetime.now().hour
            if (len(inds) > 1 or np.random.randint(3) == 0) and time < 23:
                save['epsilon'] = np.random.randint(21)
            else:
                save['epsilon'] = inds[0]
            if public['base_reward'] >= save['epsilon']:
                launch = True
                reason = 8
            else:
                launch = False
                reason = 9
        # Player model updated for optimization
        save['reason'] = reason
        with open(server+player+'.json', 'w') as json_save:
            json.dump(save, json_save)
            json_save.close()
        # Returns if the player will launch
        return launch

    def launch_tech(players, public):
        # Returns tech being launched
        o_tech = 0
        for i in players:
            with open(server+i+'.json', 'r') as json_save:
                save = json.load(json_save)
                json_save.close()
            if PlayerModel.launching(i, public, save['tech']):
                o_tech += save['tech']
        return o_tech


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
    My great strategy
    """

    # Constants
    av_reward = 11.05  # Same for both base and hidden distribution
    av_tech = 5  # Same for both auction and given tech

    def __init__(self):
        pass
        # Reads and writes file

    def tech_reward(self, x):
        return float(1.207 * np.sqrt(x))

    def bid(self, private, public):
        # game_stage = 'auction_round'

        # Assumes only one person wins the bid (typical case)
        tech_total = self.tech + self.o_tech + self.av_tech
        # Reward includes cost of launch
        reward_tech = self.tech_reward(tech_total)
        reward = self.b_reward + 6.05 + reward_tech
        # Expected value if someone else wins the bid
        e_val = reward * self.tech / tech_total
        # Find expected value if bid is won.
        e_val2 = reward * (self.tech + self.av_tech) / tech_total

        # Max possible bidding value capped at the point of no profit
        bid = round(e_val2 - e_val - 0.5)  # Floor
        # Bid just above current bidding price IF set
        if max(self.t_price) < bid - 1:
            bid = max(self.t_price) + 1

        # Minimum bid of 1 to stifle compition (max two times)
        if bid < 1 and self.a_round < 3:
            bid = 1

        # IS IT WORTH LAUNCHING NOW OR WAITING?
        launch = False

        # Average reward for each player assuming nash equalibrium
        # Also assumes only one player wins an auction every round
        bar = 22.1 / self.p_left + 5 + self.tech_reward(5 * self.p_left + 5)
        # PDF of rewards
        mu, sigma = 1.86426374483, 1.03711925603
        # Lognormal pdf twice to simulate hidden + base reward probabilities

        def logn(x):
            return (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma
                    * np.sqrt(2 * np.pi)))
        # reward of next one plus 5 lost from waiting
        chance = integrate.quad(lambda x: logn(np.sqrt(x)) / (2*np.sqrt(x)),
                                0, reward+5)
        # Simple calculation: what is the expected reward from lauching next
        e_val_nt = (reward+5) * (1 - chance[0])
        # Is it worth launching (always launch on last round)?
        # TWO CALCULATIONS
        if (self.turn == turns and e_val > 5) or (e_val > e_val_nt and reward > bar):
            launch = True

        # Desperate bankruptcy advoidance
        if self.bank < bid or (self.bank <= 5 and not launch):
            bid = self.bank - 1
            launch = True
        return bid, launch

    def join_launch(self, private, public):
        return False

    def update_status(self, private, public):
        me = private['name']
        self.tech = private['tech']  # My tech
        self.bank = private['bankroll']  # My bankroll
        self.turn = public['round']  # Round number
        self.stage = public['game_stage']
        self.a_round = public['auction_round']  # Current auction round
        # Failure "tech"
        self.f_tech = float(3.266 * np.exp(-1 * self.turn * 0.018) + 0.210)
        # Find players left in the game
        p_left = []
        for i in public['players']:
            if public['players'][i]['bankroll'] > -1 and i != me:
                p_left.append(i)
        self.p_left = len(p_left)
        # Initialize players at start of game
        if self.stage == 'start':
            # Resets other players' models.
            for i in public['players']:
                if i != me:
                    PlayerModel.initialize(i)
            self.t_price = [100, 100, 100]  # Last 3 bids from OTHERS
        # Update tech of players
        elif self.stage == 'business_done':
            self.b_reward = public['base_reward']  # Base reward
            # Give all other players +5 tech
            PlayerModel.business_done()
        elif self.stage == 'auction_round':
            # Failure tech + other launching player techs
            self.o_tech = PlayerModel.launch_tech(p_left, public)
        elif self.stage == 'auction_round_done':
            lwb = public['last_winning_bidders']
            # Give all winning other players +4 tech
            PlayerModel.auction_done(lwb)
            if me not in lwb or len(lwb) > 1:
                del self.t_price[0]
                self.t_price.append(public['last_winning_bid'])
        elif self.stage == 'launch_done':
            # Update induvidual player models
            PlayerModel.launch_done(public['last_launching_players'], public)
        elif self.stage == 'mining_done':
            # Reset launching players tech to 0
            PlayerModel.mining_done(public['last_launching_players'])
        return False

    def broadcast(self, message):
        print(message)
