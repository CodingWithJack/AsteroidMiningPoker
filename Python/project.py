import time as wait
import json
import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from paramiko import SSHClient
from scp import SCPClient
from shutil import copyfile
from scipy.optimize import curve_fit


#  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- #

# USER VARIABLES CHANGE THESE!

login = 'change_me'  # Login to the ssh serverkey
password = 'my_password'  # Password for the ssh server
my_name = 'Change_me'  # Your name in the game
games = {'prospector': 201, 'hearst': 300}  # Names of games and no. rounds
# For daily wins and get_daily logs, the start dates for each game are needed
dates = {'prospector': dt.date(2018, 10, 10), 'hearst': dt.date(2019, 1, 10)}
default = 'prospector'  # Optional name of default game for functions
run_setup = True  # If any new games are added or first time, change to True

#  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- #


print('Server commands only work for servers connected to soton.ac.uk')
# Organize information and establish framework
if default == '':
    default = list(games.keys())[0]


def get_local_log(game=default):
    """
    Downloads latest local log. This function will make sure the file is
    completed and unique.

    Returns False if the file is not complete or unique and does not copy the
    file. Otherwise, this function returns True.

    One optional string parameter: game. This changes the online directory
    from which the download occurs to /home/game/logs.

    The copied file is saved as "YYYY-MM-DD-TIME.log". The TIME component is
    minutes elapsed from 12:00AM the day the function is called.

    The file is processed with the compile_data() command and saved in the
    local directory: 'Logs/game/'+my_name.

    If the local log ended with a bankruptcy, this function calls the
    get_global_log() function to obtain the corresponding global log.
    """
    online_dir = '/home/{}/logs'.format(game)
    new_path = game.title() + '/'
    temp_dir = 'Logs/{}Temp'.format(new_path)
    local_dir = 'Logs/{}{}'.format(new_path, my_name)
    # Connects to the game server
    server = 'eldorado.clients.soton.ac.uk'
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(server, username=login, password=password)
    # Gets the current date and time
    date = dt.datetime.now().strftime('%Y-%m-%d')
    time = dt.datetime.now().hour*60 + dt.datetime.now().minute
    string_time = '-' + '0' * (4 - len(str(time))) + str(time)
    if time < 10:
        print('Operation failed: waiting for first log of the day')
        return False
    # Copies the log onto local machine
    with SCPClient(ssh.get_transport()) as scp:
        scp.get(online_dir+'/{}/{}.log'.format(my_name, date), temp_dir)
        ssh.close()  # Disconnects from the game server
    # Checks to see if file is complete and not corrupted
    with open(temp_dir+'/{}.log'.format(date)) as json_file:
        try:
            data = json.load(json_file)  # Raises a ValueError if incomplete
            parts = len(data)-1
            if parts == -1:
                raise ValueError
            else:
                stage = data[parts]['public']['game_stage']
                if not (stage == 'bankrupt' or stage == 'end'):
                    raise ValueError
            # Makes sure file isn't glitched
            turn = 0
            for i in range(len(data)):
                if data[i]['public']['round'] < turn:
                    print('Log is glitched!')
                    raise ValueError
                turn = data[i]['public']['round']
        except ValueError:
            print('Operation failed: log still being formed.')
            os.remove(temp_dir+'/{}.log'.format(date))
            return False
        json_file.close()
    # Checks player hasn't gone bankrupt!
    if data[-1]['public']['game_stage'] == 'bankrupt':
        print('Strategy went bankrupt!')
        dots = 1
        while not get_global_log(game=game, timestamp=string_time):
            print('Fetching global log' + '.' * (dots % 3 + 1))
            dots += 1
            wait.sleep(10)
    # Checks if file is unique
    all_logs = os.listdir(local_dir)
    for log in all_logs:
        if log.startswith(date):
            if abs(int(log[11:15]) - time) < 10:
                print('Operation failed: log already in directory')
                os.remove(temp_dir+'/{}.log'.format(date))
                return False
    # Moves file from holding directory into local directory
    new_path = local_dir+'/{}.log'.format(date+string_time)
    compile_data(temp_dir+'/{}.log'.format(date), new_path)
    print('New log: '+local_dir+'/{}.log'.format(date+string_time))
    return True


def get_global_log(game=default, timestamp=''):
    """
    Downloads the latest global log.

    Two optional string parameters:
    game. This changes the online directory from which the download occurs to
    /home/game/logs. The local directory to which the file is save is changed
    to Logs/game/Global.

    Timestamp. This is called by the get_local_log() function and should not
    be used otherwise. Timestamp simply overwrites the TIME component of the
    name of the downloaded log, so as to match a parallel local log.

    All copied files are saved as 'YYYY-MM-DD-TIME.game.log.' in the local
    directory: 'Logs/game/Global'.
    """
    online_dir = '/home/{}/logs'.format(game)
    new_path = game.title() + '/'
    temp_dir = 'Logs/{}Temp'.format(new_path)
    global_dir = 'Logs/{}Global'.format(new_path)
    error_print = 'Operation failed: log incomplete or too early in the day'
    if timestamp != '':
        error_print = ''
    # Connects to the game server
    server = 'eldorado.clients.soton.ac.uk'
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(server, username=login, password=password)
    # Gets the current date and time
    date = dt.datetime.now().strftime('%Y-%m-%d')
    time = dt.datetime.now().hour*60 + dt.datetime.now().minute
    if time < 10:
        print(error_print)
        return False
    # Copies the log into a temporary directory.
    with SCPClient(ssh.get_transport()) as scp:
        scp.get(online_dir+'/{}.game.log'.format(date), temp_dir)
        ssh.close()
    # Makes sure file is complete
    with open(temp_dir+'/{}.game.log'.format(date), 'r') as file:
        lines = file.readlines()
        file.close()
        if len(lines) != 0:
            last_line = lines[len(lines)-1]
            if 'Top player at end of game: ' not in last_line:
                os.remove(temp_dir+'/{}.game.log'.format(date))
                print(error_print)
                return False
        else:
            os.remove(temp_dir+'/{}.game.log'.format(date))
            print(error_print)
            return False
    # Moves file from holding directory into local directory
    if timestamp == '':
        timestamp = '-' + '0' * (4 - len(str(time))) + str(time)
    new_path = global_dir+'/{}.game.log'.format(date+timestamp)
    os.rename(temp_dir+'/{}.game.log'.format(date), new_path)
    print('New log: '+global_dir+'/{}.game.log'.format(date+timestamp))
    return True


def get_daily_logs(game=default):
    """
    Downloads all daily logs up to yesterday's date.

    One optional string parameter: game. This changes the online directory
    from which the download occurs to /home/game/logs.

    All logs are saved as 'YYYY-MM-DD.game.log' in the local directory
    Logs/game/Daily.
    """
    daily_dir = 'Logs/{}/Daily'.format(game.title())
    online_dir = '/home/{}/logs'.format(game)
    logs = [i for i in os.listdir(daily_dir) if i.endswith('.log')]
    # Lists all daily logs not yet downloaded
    start = dates[game]
    end = dt.date.today()
    delta = end - start
    for i in range(delta.days + 1):
        if str(start + dt.timedelta(i))+'.game.log' not in logs:
            start = start + dt.timedelta(i)
            break
    # Connects to the game server
    server = 'eldorado.clients.soton.ac.uk'
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(server, username=login, password=password)
    # Pulls all logs up to yesterday
    delta = end - start
    for i in range(delta.days):
        with SCPClient(ssh.get_transport()) as scp:
            log = str(start + dt.timedelta(i))
            scp.get(online_dir+'/{}.game.log'.format(log), daily_dir)
    ssh.close()  # Disconnects from the game server


def auto_log(game=default):
    """
    Runs the get_local_log() function every 10 minutes continuously.

    One optional string parameter: game. This changes the game the function
    takes logs from by calling get_local_log(game).

    If a log cannot be downloaded, the function waits 1 minute and tries again.
    A statement is printed for every 10 logs downloaded.
    """
    n = 0
    while True:
        while not get_local_log(game=game):
            wait.sleep(60)
        n += 1
        if n % 10 == 0 and n != 0:
            print('Logs harvested: '+str(n))
        wait.sleep(60 * 10)


def get_file(name):
    """
    Downloads a file from the client server.

    One string parameter: name. This parameter must be set to the path name of
    the file required.

    Files are saved in the current working directory folder 'Downloads'.
    """
    # Connects to the game server
    server = 'eldorado.clients.soton.ac.uk'
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(server, username=login, password=password)
    # Pulls all logs up to yesterday
    with SCPClient(ssh.get_transport()) as scp:
        scp.get(name, 'Downloads')
    ssh.close()  # Disconnects from the game server


def upload_strategy(name='strategies.py'):
    """
    Uploads a file to the online server.

    One optional string parameter: name. This should be set to the name of the
    file to be uploaded. Set to 'strategies.py' by default.

    The function also copies the file into the Journal directory and saves it
    as 'strategies-YYYY-MM-DD-TIME.py'.
    """
    # Connects to the game server
    server = 'eldorado.clients.soton.ac.uk'
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(server, username=login, password=password)
    # Copies file into ssh directory
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(name, 'strategies.py')
        ssh.close()  # Disconnects from the game server
    # Adds new strategy to journal
    date = dt.datetime.now().strftime('%Y-%m-%d')
    time = dt.datetime.now().hour*60 + dt.datetime.now().minute
    timestamp = '-' + '0' * (4 - len(str(time))) + str(time)
    copyfile(name, 'Journal/{}-{}.py'.format(name[:-3], date+timestamp))


def table(x, count=False, start=0):
    """
    Creates and returns a table containing x items in a list.

    By default, each item is equal to 0.

    This function has one required parameter: int x.

    Int x determines the length of the list.

    Two optional parameters: bool count, and int start.
    Start usually determines what every value in the table is, if count is
    set to True however, start determines the first value in the table.

    Count, if set to True makes every item in the list equal to the previous
    item in the list + 1.
    """
    table = []
    for i in range(x):
        if count is True:
            table.append(i+start)
        else:
            table.append(start)
    return table


def dictionary(x, index):
    """
    Creates and returns a dictionary with x members. Each member is assigned
    an index, which it is set to return if called on by the dictionary.

    Two parameters: list x, and (any type) index.
    List x is the list of keys the dictionary contains.

    Index is what each item in the list matches to within the dictionary.
    """
    my_dict = {}
    for i in x:
        my_dict[i] = index
    return my_dict


def create_clean_json():
    """
    Creates or replaces a json file named 'compiled_data.json' in the 'Logs'
    directory.

    The json file contains the framework data the compile_data function
    requires to operate.

    Upon being called, this function asks a safety question as it has the
    potential to overwrite valuable data.
    """
    safety = input("""
    Are you sure you want to continue? This will delete any existing json
    files. [y/n]
    """)
    if safety[0].lower() == 'y':
        data = {'asteroid_base': [],
                'asteroid_unseen': [],
                'tech_base': [],
                'tech_auction': [],
                'tech_function': [],
                'rounds': dictionary(list(range(1, 202)), {})}
        with open('Logs/compiled_data.json', 'w') as outfile:
            json.dump(data, outfile)
            outfile.close()


def compile_data(log, move_to):
    """
    Compiles and saves important data from a local log. The log is then moved
    to a new location.

    All relavent data is saved in compiled_data.json' in the 'Logs' directory.

    Two string parameters: log and move_to.
    Log should be set to the path name of the log to be compiled. This can be
    set to a directory in order to compile multiple logs.

    Move_to should be set to the path name of the log once it has been
    compiled. This should be a directory path if log is also.
    """
    # Opens total compiled data
    with open('Logs/compiled_data.json', 'r') as json_save:
        save = json.load(json_save)
        json_save.close()
    ab = save['asteroid_base']
    au = save['asteroid_unseen']
    tb = save['tech_base']
    ta = save['tech_auction']
    tf = save['tech_function']
    rounds = save['rounds']
    # Opens logs and saves their data
    name_type = 0
    if log.endswith('.log'):
        logs = [log]
    else:
        logs = [(log + '/' + i) for i in os.listdir(log) if i.endswith('.log')]
        name_type = len(log) + 1
    for x in logs:
        with open(x) as json_file:
            data = json.load(json_file)
            json_file.close()
        # Adds relavent information from log to total compiled data
        for i in range(len(data)):
            if data[i]['public']['game_stage'] == 'start':
                players = dictionary(list(data[i]['public']['players']), 0)
            elif data[i]['public']['game_stage'] == 'business_done':
                # Asteroid base distribution
                diff = data[i]['public']['base_reward'] - len(ab)
                if diff >= 0:
                    ab.extend(table(diff + 1))
                ab[data[i]['public']['base_reward']] += 1
                # Base tech distribution
                new_tech = data[i]['private']['tech']
                old_tech = data[i-1]['private']['tech']
                diff = new_tech - old_tech - len(tb)
                if diff >= 0:
                    tb.extend(table(diff + 1))
                tb[new_tech-old_tech] += 1
                # Guess tech for players
                for o in players:
                    players[o] += 5
            elif data[i]['public']['game_stage'] == 'auction_round_done':
                if my_name in data[i]['public']['last_winning_bidders']:
                    # Auction tech distribution
                    new_tech = data[i]['private']['tech']
                    old_tech = data[i-1]['private']['tech']
                    diff = new_tech - old_tech - len(ta)
                    if diff >= 0:
                        ta.extend(table(diff + 1))
                    ta[new_tech-old_tech] += 1
                # Guess tech for players again
                for o in data[i]['public']['last_winning_bidders']:
                    if o in players:
                        players[o] += 5
            elif data[i]['public']['game_stage'] == 'mining_done':
                # Guess total tech spent
                tech_spent = 0
                for o in data[i]['public']['last_launching_players']:
                    if o == my_name:
                        tech_spent += data[i-1]['private']['tech']
                        players[my_name] = 0
                    else:
                        tech_spent += players[o]
                        players[o] = 0
                # Adds unseen value data to list
                total_value = data[i]['public']['last_mining_payoff']
                base_value = data[i]['public']['base_reward']
                u_val = total_value - base_value  # unseen value
                if tech_spent == 20:
                    if (u_val - len(au)) >= 0:
                        au.extend(table((u_val - len(au)) + 1))
                    au[u_val] += 1
                # Adds mining failure data to list
                turn = str(data[i]['public']['round'])
                fail_tag = 'Mission failure'
                if turn not in rounds:
                    rounds[turn] = {}
                if str(tech_spent) in rounds[turn]:
                    rounds[turn][str(tech_spent)]['frequency'] += 1
                else:
                    rounds[turn][str(tech_spent)] = {'frequency': 1, 'fail': 0}
                if data[i]['public']['last_winning_miner'] == fail_tag:
                    rounds[turn][str(tech_spent)]['fail'] += 1
                # Adds tech function data to list
                # NOTE: can be improved, include failure tech contribution?
                if len(tf)-tech_spent*2 <= 0:
                    tf.extend(table(tech_spent*2 - len(tf) + 2))
                num = u_val + tf[tech_spent*2] * tf[tech_spent*2 + 1]
                den = tf[tech_spent*2 + 1] + 1
                tf[tech_spent*2] = num / den
                tf[tech_spent*2 + 1] += 1
        # Moves compiled log elsewhere
        if name_type != 0:
            os.rename(x, move_to + '/' + x[name_type:])
        else:
            os.rename(x, move_to)
    # Saves all data
    save['asteroid_base'] = ab
    save['asteroid_unseen'] = au
    save['tech_base'] = tb
    save['tech_auction'] = ta
    save['tech_function'] = tf
    save['rounds'] = rounds
    with open('Logs/compiled_data.json', 'w') as json_save:
        json.dump(save, json_save)
        json_save.close()


def plot(ydata, xdata=None, normalize=False, bar=False, title=None, label=None,
         xlabel=None, ylabel=None, axis=None, xlim=None, ylim=None, color=None,
         logplot=False, fit=None, save=None, show=False):
    """
    Plots a graph.

    Required parameters:
        ydata: This can either be a list of data values or a string. If this
        is a string, the ydata values will be taken from a json file named
        Logs/compiled_data.json in the local directory. The function will look
        for an index in the file equivalent to this string.

    Optional parameters:
        xdata: takes a list input. The function maps the ydata onto this.
        By default, axis will be equivalent to a list of equal length to the
        ydata, counting up from 0.

        normalize: boolean. If set to True, all ydata values will be such that
        the sum is equivalent to 1.

        bar: boolean. If set to True, this function will plot the data as a
        bar chart. By default, the function plots a line graph.

        title: string. The displayed title on the chart.

        label: string. adds a legend to the chart and labels the
        data accordingly.

        xlabel: string. The displayed x axis label.

        ylabel: string: The displayed y axis label.

        axis: takes a list input length four. If given, the plot will only
        display data within this set [xmin, xmax, ymin, ymax].

        xlim: a length two tuple. Simply displays all data between the limits
        for all x data.

        ylim: a length two tuple. Simply displays all data between the limits
        for all y data.

        color: string. Changes the color of the plot to any matplotlib
        recognised color.

        logplot: boolean. If set to true, the function will fit a lognormal
        plot to the data. It will also print the mean, varience, and standard
        deviation of said data. The lognormal plot will be overlayed.

        fit: takes a function. The function will be fitted to the data and
        overlayed onto the plot.

        save: string. If set, the data will be saved as 'Analysis/INPUT' in
        the current working directory.

        show: boolean. If set to true, the plot will be displayed instantly.
        by default, this will occur at the end of processes, allowing overlays.
    """
    if type(ydata) == str:
        with open('Logs/compiled_data.json', 'r') as json_save:
            all_data = json.load(json_save)
            ydata = all_data[ydata]
            json_save.close()
    if normalize:
        total = sum(ydata)
        y_axis = []
        for i in ydata:
            y_axis.append(i / total)
        print('Data points: '+str(total))
    else:
        y_axis = ydata
    if type(xdata) == list:
        x_axis = xdata
    else:
        x_axis = table(len(y_axis), True)
    if bar:
        plt.bar(x_axis, y_axis, color=color, align='center', label=label)
    else:
        plt.plot(x_axis, y_axis, color=color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if title is not None:
        plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if label is not None:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if type(axis) == list:
        plt.axis(axis)
    if logplot:
        # Find the mean and varience
        mean = 0
        var = 0
        for i in range(len(y_axis)):
            mean += i * y_axis[i]
        for i in range(len(y_axis)):
            var += (i - mean)**2 * y_axis[i]
        print('Mean: '+str(mean))
        print('Varience: '+str(var)+'\nStandard deviation: '+str(var**0.5))
        mu = np.log(mean/(1 + var/mean**2)**0.5)
        sigma = (np.log(1 + var/mean**2))**0.5
        print(mu, sigma)
        x = np.linspace(0, axis[1], 10000)
        divide_factor = (x * sigma * np.sqrt(2 * np.pi))
        pdf = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / divide_factor
        plt.plot(x, pdf, linewidth=1.5, color='r',
                 label='lognormal fit: mu=%5.3f, sigma=%5.3f' % (mu, sigma))
        plt.legend()
    if fit:
        # Find a function that fits
        x_axis = np.array(xdata)
        popt, pcov = curve_fit(fit, x_axis, y_axis)
        tag = 'fit: '
        for i in range(len(tuple(popt))):
            tag += '{}=%5.3f'.format(chr(97+i))
            if i != len(tuple(popt))-1:
                tag += ', '
        plt.plot(x_axis, fit(x_axis, *popt), 'r-', label=tag % tuple(popt))
        plt.legend(loc='best')
    if type(save) == str:
        plt.savefig('Analysis/'+save+'.pdf', format='pdf')
    if show:
        plt.show()


# Finding base reward distribution
def base_dist():
    """
    Plots and displays a graph of the probability function of the base value
    of the asteroids.
    Graph is saved in local directory as 'base_distribution.pdf'
    """
    plot('asteroid_base', normalize=True, color='c', save='base_distribution',
         title='Base Value Distribution', xlabel='Base Value', bar=True,
         ylabel='Probability', axis=[0, 60, 0, 0.11], logplot=True)


# Finding hidden reward distribution
def hidden_dist():
    """
    Plots and displays a graph of the probability function of the hidden value
    of the asteroids.
    Graph is saved in local directory as 'hidden_distribution.pdf'
    """
    with open('Logs/compiled_data.json', 'r') as json_save:
        save = json.load(json_save)
        data = save['asteroid_unseen']
        json_save.close()
    # Uncomment the following to see an untampered unseen dirstibution graph
    """
    plot(data, normalize=True, bar=True, xlabel='Unseen Reward',
         title='Unseen Reward Distribution at 20 Tech Played',
         ylabel='Probability', show=True, axis=[0, 40, 0, 0.11],
         save='hidden_distribution_non', color='limegreen')
    """
    del data[:5]
    plot(data, normalize=True, bar=True, xlabel='Hidden Reward', logplot=True,
         title='Hidden Reward Distribution', ylabel='Probability',
         axis=[0, 40, 0, 0.11], save='hidden_distribution', color='limegreen')


# Base tech each round
def base_tech():
    """
    Plots and displays a graph of the probability distribution of tech gained
    at the start of each round.
    Graph is saved in local directory as 'base_tech.pdf'
    """
    plot('tech_base', normalize=True, title='Base Tech Distribution',
         xlabel='Tech Given', ylabel='Probability', axis=[-1, 11, 0, 0.1],
         save='base_tech', color='c', bar=True)


# Auction tech won
def auction_tech():
    """
    Plots and displays a graph of the probability distribution of tech gained
    after winning an auction.
    Graph is saved in local directory as 'auction_tech.pdf'
    """
    plot('tech_auction', normalize=True, bar=True, color='limegreen',
         title='Auction Tech Distribution', xlabel='Tech Won',
         ylabel='Probability', axis=[-1, 11, 0, 0.1], save='auction_tech')


def tech_reward(x, a, b):
    """
    Guess function for the tech reward, where x is the tech submitted.
    Takes three parameters, all floats or integers: a, b, and x.
    Returns a * sqrt(x) + b, a numpy float.
    """
    return a * np.sqrt(x) + b


def tech_function(resolution=20):
    """
    Plots a graph showing the change of the average reward against the tech
    launched by all players.

    Has one optional integer parameter: resolution. This the minimum amount of
    data a tech value needs to be plotted. By default this is set to 20.
    For example, the average reward observed from tech value which only has 19
    observations will not be plotted

    This function also plots a graph showing the error in each measurement.
    Error is calculated using standard deviation divided by the square root of
    the number of observations.
    """
    with open('Logs/compiled_data.json', 'r') as json_save:
        save = json.load(json_save)
        data = save['tech_function']
        json_save.close()
    xdata, ydata, yerror, xerror = [], [], [], []
    for i in range(len(data)):
        if 2*i+1 < len(data):
            for o in range(data[2*i+1]):
                if data[2*i+1] > resolution:
                    ydata.append(data[2*i])
                    xdata.append(i)
            if data[2*i+1] > resolution:
                yerror.append(15.4 / data[2*i+1]**0.5)
                xerror.append(i)
        else:
            break
    plot(ydata, xdata, color='b', xlabel='Tech Played', fit=tech_reward,
         title='Variation of Average Unseen Value with Tech Played',
         ylabel='Average Unseen Value', show=True, save='tech_function')
    plot(yerror, xerror, color='g', ylabel='Standard Deviations of Error',
         xlabel='Tech Played', show=True, save='tech_function_error')


def round_failure(x, return_tech=False):
    """
    Plots and displays a graph showing the distribution of tech spent by all
    players for a single round. The graph is overlayed by another bar chart in
    red, showing - for all of these rounds - which ones ended in a Mission
    Failure win.

    Function has one int parameter x, which must be set equal to the round
    wished analysed.

    One optional bool parameter, return_tech. If this is set to True, the
    function calculates and returns the approximate value that the
    'Mission Failure' AI spent on tech in the specified round.
    """
    with open('Logs/compiled_data.json', 'r') as json_save:
        save = json.load(json_save)
        run = save['rounds'][str(x)]
        json_save.close()
    plays, fails = [], []
    for i in run:
        if int(i) >= len(plays):
            plays.extend(table(int(i)+1 - len(plays)))
            fails.extend(table(int(i)+1 - len(fails)))
        plays[int(i)] += run[i]['frequency']
        fails[int(i)] += run[i]['fail']
    if not return_tech:
        plot(plays, bar=True, color='b', label='Games Played')
        plot(fails, bar=True, color='r', ylabel='Frequency', show=True,
             title='Average Tech Played with Failure Rate for Round '+str(x),
             xlabel='Tech Played', label='Failure Wins')
    else:
        f_tech, weight = 0, 0
        for i in range(len(plays)):
            if plays[i] > fails[i] and fails[i] > 0:
                tech = i * fails[i]
                freq = plays[i]
                f_tech = (tech + f_tech * weight) / (weight + freq)
                weight += freq
        return f_tech


def fail_submission(x, a, b, c):
    """
    Guess function for the failure tech submitted, where x is the round.
    Takes four parameters, all floats or integers: a, b, c, and x.
    Returns a * e^(-1*x*b) + c, a numpy float.
    """
    return a * np.exp(-1 * x * b) + c


# Most likely failure tech per round
def failure_tech():
    """
    Plots and displays a graph of the estimated 'Mission Failure' tech value
    for all 200 rounds.
    Graph is saved in local directory as 'mission_failure_tech.pdf'.
    """
    x_axis = np.linspace(1, 201, 201)
    y_axis = []
    for i in x_axis:
        y_axis.append(round_failure(int(i), return_tech=True))
    plot(y_axis, list(x_axis), xlabel='Round', ylabel='Tech Submitted',
         title="Tech Submitted by 'mission_failure' per Round", color='b',
         save='mission_failure_tech', fit=fail_submission, axis=[1, 201, 0, 4])


def daily_wins(show=[], cumulative=False, start=-10, game=default):
    """
    Plots a graph showing the final bankrolls from the Daily Logs folder. By
    default, it will plot the bankrolls for the past 10 days.

    Four optional parameters:
    List, show. If this is filled with the names of players, the
    graph will only plot the final bankrolls for those players

    Bool, cumulative. If set to True the graph will show cumulative bankrolls
    added up from the beginning of counting.

    Int, start. This determines the start day on the x-axis (counting from 0),
    if set to a negative number, it will show the past n days where n is the
    absolute value of start.

    String, game. Determines the game for which the bankrolls are plotted.
    """
    get_daily_logs(game=game)
    daily_dir = 'Logs/{}/Daily'.format(game.title())
    logs = [i for i in os.listdir(daily_dir) if i.endswith('.log')]
    start_day = dates[game]
    players = {}
    for x in logs:
        log_day = dt.datetime.strptime(x[:-9], "%Y-%m-%d").date()
        position = (log_day - start_day).days
        with open('{}/{}'.format(daily_dir, x)) as file:
            data = file.readlines()
            file.close()
        i = 0
        while data[-2-i][:5] == 'After':
            words = data[-2-i].split()
            player = ''
            for o in range(len(words) - 9):
                if o != 0:
                    player += ' '
                player += words[o+4]
            if player not in players:
                players[player] = table(len(logs))
            if int(words[-1][:-1]) > 0:
                players[player][position] = int(words[-1][:-1])
            i += 1
    title = 'Winning charts'
    ylabel = 'End game bankroll'
    if cumulative:
        title = 'Cumulative winning charts'
        ylabel = 'Total bankroll'
        for i in players:
            for o in range(len(players[i])):
                if o != 0:
                    players[i][o] += players[i][o-1]
    xlim = 0 + start
    if start < 0:
        xlim = len(logs) + start
    for i in players:
        if show == [] or i in show:
            plot(players[i], title=title, label=i, xlabel='Day', ylabel=ylabel,
                 color=p_colors(i), xlim=(xlim, len(logs)-1))
    # plt.savefig('cumulative_bank.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def plot_bankroll(log):
    """
    Plots the variation of bankroll for the beginning of each round across a
    game.

    One string parameter is required: log. Log must be set to the name of a
    global log file (any.game.log) in the current working directory.
    """
    # Reads the data from the log
    with open(log, 'r') as file:
        lines = file.readlines()
        file.close()
    # Sets up data to save
    total_turns = 1000
    players = {}
    turn = -1

    def get_player(line):
        # Finds and returns the player concerned in the line given
        for i in players:
            if i in line:
                return i
    # Goes through global log and saves bank data
    for i in range(len(lines)):
        if '-*-*-*-*-' in lines[i]:
            turn += 1
        if lines[i].endswith('has 1000 money.\n') and turn == 0:
            new_player = lines[i][:len(lines[i])-17]
            players[new_player] = [1000] + list(np.linspace(0, 0, 1000))
        elif lines[i].endswith(' money.\n'):
            player = get_player(lines[i])
            bankroll = int(lines[i].split()[-2])
            if bankroll > 0:
                players[player][turn] = bankroll
        elif lines[i].startswith('After '):
            player = get_player(lines[i])
            bankroll = int(lines[i].split()[-1][:-1])
            if bankroll > 0:
                players[player][turn] = bankroll
            total_turns = int(lines[i].split()[1])
    # Plot bankroll graph
    for i in players:
        x_axis = list(np.linspace(0, total_turns, total_turns+1))
        plt.plot(x_axis, [1000]+players[i][:total_turns], label=i,
                 color=p_colors(i))
    plt.xlabel('Turn'), plt.ylabel('Bankroll'), plt.xlim(0, total_turns)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)), plt.grid()


def p_colors(players):
    """
    Returns the colour for a player inputed, taken from a pre-set database.
    The function takes one parameter which can either be a string or a list of
    strings: 'name'.

    If a players name is not recognised, a random colour for that player will
    be returned. Returns either a string, or a list of strings.
    """
    player_colors = {'Peploe': 'm', 'Freckelton': 'g', 'Simpson': 'c',
                     my_name: 'r', 'Comerford': 'limegreen',
                     'Lovelock': 'orange', 'Evie': 'b', 'Wittig': 'seagreen',
                     'Mission failure': 'maroon', 'Random': 'palevioletred',
                     'Jupiter Mining Corp': 'skyblue'}
    colors = []
    return_string = False
    if type(players) == str:
        players = [players]
        return_string = True
    for i in players:
        if i in player_colors:
            colors.append(player_colors[i])
        else:
            r = 1 - 0.01 * np.random.randint(101)
            g = 1 - 0.01 * np.random.randint(101)
            b = 1 - 0.01 * np.random.randint(101)
            colors.append((r, g, b))
    if return_string:
        return colors[0]
    else:
        return colors


# Add returns for function below
def analyse(log, game=default, show=[], r_data=False):
    """
    Plots various graphs associated with a log. In order of appearence these
    are: tech variation per round for all players, bankroll per round for all
    players, base value and auction price of asteroids per round, percent of
    auctions won per player, money spent of auctions per player, fraction of
    tech committed per player, fraction of asteroids mined per player, and
    fraction of final bankroll per player.

    If a local log goes bankrupt, the function will terminate and instead run
    plot_bankroll() for the corresponding global log (if it exists).

    Function has one necessary string parameter: log. This should be set to
    the name of a log wished analyse in the format YYYY-MM-DD-TTTT.log.

    Three optional parameters:
    String, game. This is the game for which the log was recorded.

    List, show. If set to a list of players, data will only be shown for those
    players.

    Bool, r_data. If set to True, the function will return data in the form of
    a tuple. No graphs will be plotted.
    """
    with open('Logs/{}/{}/{}'.format(game.title(), my_name, log)) as json_file:
        data = json.load(json_file)
        json_file.close()
    players = list(data[0]['public']['players'])
    if show != []:
        players = list(set(show) & set(players))
    # Check to see if can analyse all players
    if len(players) == 0:
        return False
    # Check to see if log went bankrupt & print bankruptcy data
    if data[-1]['public']['game_stage'] == 'bankrupt':
        print('Player went bankrupt!')
        print('Bankruptcy round: '+str(data[-1]['public']['round']))
        global_log = 'Logs/{}/Global/{}game.log'.format(game.title(), log[:-3])
        try:
            with open(global_log, 'r') as file:
                lines = file.readlines()
                file.close()
            for u in range(len(lines)):
                start = u
                if '-*-*-*-*-' in lines[u]:
                    break
            for o in players:
                for u in range(len(lines)-1, -1, -1):
                    if o in lines[u]:
                        if (lines[u].startswith(o+' is bankrupt') or
                           lines[u].startswith('After')):
                            print(lines[u][:-2])
                            break
                    if u <= start:
                        print(o+' is bankrupt in round 0')
                        break
            # Plots the bankroll variation for the global log
            plot_bankroll(global_log)
        except FileNotFoundError:
            print('No global log for '+log)
        return
    # Find per player stats
    tech_variation, bank_variation, tech_commited = {}, {}, {}
    auctions_won, auction_cost, asteroids_mined = {}, {}, {}
    total_auctions, failed_asteroids = 0, 0
    for o in players:
        tech = list(np.linspace(0, 0, num=(1+data[-1]['public']['round']*2)))
        bank = [1000]
        auctions = 0
        money_spent = 0
        asteroids = 0
        t_commited = 0
        for i in range(len(data)):
            turn = data[i]['public']['round'] - 1
            bankroll = data[i]['public']['players'][o]['bankroll']
            stage = data[i]['public']['game_stage']
            if bankroll > 0:
                if stage == 'business_done':
                    tech[1 + 2*turn] += 5
                elif stage == 'auction_round_done':
                    if o in data[i]['public']['last_winning_bidders']:
                        tech[1 + 2*turn] += 5
                        auctions += 1
                        money_spent += data[i]['public']['last_winning_bid']
                    if o == my_name:
                        total_auctions += 1
                elif stage == 'launch_done':
                    bank.append(bankroll)
                elif stage == 'mining_done':
                    if o in data[i]['public']['last_launching_players']:
                        t_commited += tech[1 + 2*turn]
                        tech[2 + 2*turn] = 0
                    else:
                        tech[2 + 2*turn] = tech[1 + 2*turn]
                        if turn != games[game]-1:
                            tech[3 + 2*turn] = tech[1 + 2*turn]
                    bank.append(bankroll)
                    if o in data[i]['public']['last_winning_miner']:
                        asteroids += 1
                    elif (o == my_name and 'Mission failure' in
                          data[i]['public']['last_winning_miner']):
                        failed_asteroids += 1
                if o == my_name and stage == 'launch_done':
                    tech[1 + 2*turn] = data[i]['private']['tech']
            elif (stage == 'launch_done' or stage == 'mining_done'):
                bank.append(0)
        tech_variation[o] = tech
        bank_variation[o] = bank
        auctions_won[o] = auctions
        auction_cost[o] = money_spent
        if asteroids > 0:
            asteroids_mined[o] = asteroids
        if t_commited > 0:
            tech_commited[o] = t_commited
        # Returns values for macro analysis
        if r_data == o:
            return (bank, asteroids, t_commited, failed_asteroids)
    asteroids_mined['Mission failure'] = failed_asteroids
    # Find tech price stats
    t_price, b_val, n_val = [], [], []
    for i in range(len(data)):
        if data[i]['public']['game_stage'] == 'mining_done':
            lwb = data[i]['public']['last_winning_bid']
            t_price.append(lwb)
            b_val.append(data[i]['public']['base_reward'])
            if lwb > 0:
                n_val.append(data[i]['public']['base_reward'] / lwb)
            else:
                n_val.append(1000)  # A big number
    turns = data[-1]['public']['round']
    # Plot tech variation graphs
    for i in players:
        x_axis = [0] + list(np.linspace(1, turns+0.5, num=turns*2))
        plot(tech_variation[i], x_axis, xlabel='Round', ylabel='Tech',
             title=i+"'s tech variation across game: "+log,
             xlim=(0, turns+0.5), show=True, color=p_colors(i))
    # Plot bankroll graph
    for i in players:
        x_axis = [0] + list(np.linspace(1, turns+0.5, num=turns*2))
        plot(bank_variation[i], x_axis, xlabel='Round', ylabel='Bankroll',
             title='Bank variation across game: '+log, label=i,
             xlim=(0, turns+0.5), color=p_colors(i))
    plt.ylim(bottom=0)
    plt.show()
    # Plot auction prices
    x_axis = table(turns, count=True, start=1)
    plot(t_price, x_axis, label='Auction Value')
    plot(b_val, x_axis, label='Base Value')
    plot(n_val, x_axis, label='Normalized Value', xlabel='Round',
         ylabel='Price', xlim=(1, turns), show=True,
         title='Tech price variation across game: '+log)
    # Plot percent of auctions won bar chart
    pos = 1
    for i in auctions_won:
        auctions_won[i] = 100 * auctions_won[i] / total_auctions
        plt.barh([pos], auctions_won[i], align='center', label=i,
                 color=p_colors(i))
        pos += 1
    plt.yticks(np.arange(1, pos), list(auction_cost))
    plt.xlabel('Auctions won (%)')
    plt.title('Percentage of auctions won across game: '+log)
    for i, v in enumerate(auctions_won.values()):
        plt.text(v+1, i + 0.875, '%5.2f' % v)
    plt.xlim(0, max(auctions_won.values())*1.14)
    plt.grid()
    plt.show()
    # Plot price payed on auctions
    pos = 1
    for i in auction_cost:
        plt.barh([pos], auction_cost[i], align='center', label=i,
                 color=p_colors(i))
        pos += 1
    plt.yticks(np.arange(1, pos), list(auction_cost))
    plt.xlabel('Money Spent')
    plt.title('Bankroll commited to auctions across game: '+log)
    for i, v in enumerate(auction_cost.values()):
        plt.text(v+1, i + 0.875, '%5.0f' % v)
    plt.xlim(0, max(auction_cost.values())*1.13)
    plt.grid()
    plt.show()
    # Plot tech commited pie chart
    plt.pie(list(tech_commited.values()), colors=p_colors(list(tech_commited)),
            labels=list(tech_commited), autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution showing tech commited by each player\nacross ' +
              'game: '+log, y=1.1)
    plt.show()
    # Plot percent of asteroids mined
    plt.pie(list(asteroids_mined.values()), labels=list(asteroids_mined),
            colors=p_colors(list(asteroids_mined)), autopct='%1.1f%%',
            startangle=90)
    plt.axis('equal')
    plt.title('Distribution showing fraction of asteroids mined by\n' +
              'each player across game: '+log, y=1.1)
    plt.show()
    # Plot final bankroll pie chart
    bank_endings = {}
    for i in bank_variation:
        if bank_variation[i][-1] > 0:
            bank_endings[i] = bank_variation[i][-1]
    plt.pie(list(bank_endings.values()), colors=p_colors(list(bank_endings)),
            labels=list(bank_endings), autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution showing final bankroll of each player\nacross ' +
              'game: '+log, y=1.1)
    plt.show()


def macro_analysis(n, game=default, show=[]):
    """
    Plots various graphs associated with the past n logs. In order of
    appearence these are: average bankroll per round for all players, average
    fraction of tech committed per player, average fraction of asteroids mined
    per player, and average fraction of final bankroll per player.

    The function takes the most recent log and counts backwards in time to get
    a sample of logs with a size of n.

    This function will skip any bankrupt logs. It will also print how many
    times each player went bankrupt in the sample given, as well as printing
    their average bankrupt round if they did.

    Function has one necessary int parameter: n. This should be set to the
    number of past logs to include in the sample.

    Two optional parameters:
    String, game. This is the game for which sample is taken.

    List, show. If set to a list of players, data will only be shown for those
    players.
    """
    # Make list of last n logs
    local = 'Logs/{}/{}'.format(game.title(), my_name)
    all_logs = [i for i in os.listdir(local) if i.endswith('.log')]
    if n > len(all_logs):
        print("Too many logs requested!")
        return
    delta = 0
    logs = []
    while len(logs) < n:
        date = str(dt.date.today() - dt.timedelta(delta))
        delta += 1
        new_logs = [i for i in os.listdir(local) if i.startswith(str(date))]
        if len(new_logs) <= n - len(logs):
            logs += new_logs
        else:
            for i in range(1440, -1, -1):
                time = '-' + '0' * (4 - len(str(i))) + str(i) + '.log'
                if str(date)+time in new_logs:
                    logs.append(str(date)+time)
                if len(logs) == n:
                    break
    # Make dictionary of all players involved and get number of rounds
    players = []
    bankrupt_logs = []
    for i in logs:
        with open(local+'/'+i) as json_file:
            data = json.load(json_file)
            json_file.close()
        # Adds players to a big list of unique players
        for o in list(data[0]['public']['players']):
            if o not in players:
                players.append(o)
        # Adds bankrupt logs to database
        if data[-1]['public']['game_stage'] == 'bankrupt':
            bankrupt_logs.append(i)
            print('bankrupt log: '+i)
        else:
            # Gets number of turns in the logs
            turns = data[-1]['public']['round']
    if show != []:
        players = list(set(players) & set(show))
    # Prepare bankruptcy data
    b_rounds = dictionary(players, 0)
    b_times = dictionary(players, 0)
    for i in bankrupt_logs:
        global_log = 'Logs/{}/Global/{}game.log'.format(game.title(), i[:-3])
        try:
            with open(global_log, 'r') as file:
                lines = file.readlines()
                file.close()
            for u in range(len(lines)):
                start = u
                if '-*-*-*-*-' in lines[u]:
                    break
            for o in players:
                for u in range(len(lines)-1, -1, -1):
                    if o in lines[u]:
                        if lines[u].startswith(o+' is bankrupt'):
                            b_rounds[o] += int(lines[u][:-2].split()[-1])
                            b_times[o] += 1
                            break
                        elif u <= start:
                            b_times[o] += 1
                            break
        except FileNotFoundError:
            print('No global log for '+i)
        logs.remove(i)
    # Get data and plot average bankroll per round
    tech_commited = dictionary(players, 0)
    total_mined = dictionary(players, 0)
    bank_total = dictionary(players, 0)
    total_mined['Mission failure'] = 0
    x_axis = [0] + list(np.linspace(1, turns+0.5, num=turns*2))
    for o in players:
        bankroll = [0] + list(np.linspace(1, turns+0.5, num=turns*2))
        skipped_logs = 0
        for x in range(len(logs)):
            p_data = analyse(logs[x], show=[o], game=game, r_data=o)
            if type(p_data) == bool:
                skipped_logs += 1
                continue
            la = x - skipped_logs  # Logs analysed
            bank, asteroids, t_com, fails = p_data
            total_mined[o] += asteroids
            total_mined['Mission failure'] += fails
            tech_commited[o] += t_com
            bank_total[o] += bank[-1]
            for i in range(len(bank)):
                bankroll[i] = (bank[i] + bankroll[i] * la) / (la + 1)
                if i == len(bank) - 1 and bank[i] == 0:
                    b_times[o] += 1
                    b_rounds[o] += int((bank.index(0)+1)/2)
        if bank_total[o] == 0:
            del bank_total[o]
        if tech_commited[o] == 0:
            del tech_commited[o]
        if total_mined[o] == 0:
            del total_mined[o]
        plot(bankroll, x_axis, xlabel='Round', ylabel='Average Bankroll',
             title='Bank variation across past {} games'.format(n), label=o,
             xlim=(0, turns+0.5), color=p_colors(o))
    plt.ylim(bottom=0)
    plt.show()
    # Plot tech commited pie chart
    plt.pie(list(tech_commited.values()), colors=p_colors(list(tech_commited)),
            labels=list(tech_commited), autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution showing average distribution of tech\n commited' +
              ' by each player across past {} games'.format(n), y=1.1)
    plt.show()
    # Plot asteroids mined
    plt.pie(list(total_mined.values()), colors=p_colors(list(total_mined)),
            labels=list(total_mined), autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution showing fraction of asteroids mined by\n' +
              'each player across past {} games'.format(n), y=1.1)
    plt.show()
    # Pie final bankroll
    plt.pie(list(bank_total.values()), labels=list(bank_total),
            autopct='%1.1f%%', startangle=90,
            colors=p_colors(list(bank_total)))
    plt.axis('equal')
    plt.title('Distribution average final bankroll for\n' +
              'each player across past {} games'.format(n), y=1.1)
    plt.show()
    # Print bankruptcy data
    print('\n')
    for i in players:
        text = i+' had {} bankruptcies'.format(b_times[i])
        if b_times[i] > 0:
            text += ' with an average bankruptcy round of %5.2f' % (
                     b_rounds[i] / b_times[i])
        print(text)
    # Print out bankruptcy statistics & average bankrupt round.
    print('\nlogs skipped: ' + str(len(bankrupt_logs)))


# Runs setup if needed
if run_setup:
    if not os.path.isdir('Logs'):
        os.makedirs('Logs')
        create_clean_json()
    for i in list(games.keys()):
        if not os.path.isdir('Logs/'+i.title()):
            os.makedirs('Logs/'+i.title())
            os.makedirs('Logs/'+i.title()+'/Temp')
            os.makedirs('Logs/'+i.title()+'/Daily')
            os.makedirs('Logs/'+i.title()+'/'+my_name)
            os.makedirs('Logs/'+i.title()+'/Global')
