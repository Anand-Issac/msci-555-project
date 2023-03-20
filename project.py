import copy
import math
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from dateutil import parser
import random

"""
HELPER FUNCTIONS
"""

"""
Returns great-circle distance between two points on globe
"""
#taken directly from https://stackoverflow.com/a/4913653 
# helper function to calculate distances between arenas
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

"""
Builds a mapping between a team and all of their games
"""
def create_team_schedule(schedule):
    team_schedule = {}

    # build a map between team and all of their games
    for i in range(len(teams)):
        team = teams[i]
        curr = []
        for j in range(len(schedule)):
            game = schedule[j]
            if team in game:
                curr.append(game)
        team_schedule[team] = curr
    return team_schedule

"""
Calculates the total distance travelled by all teams in the given schedule

logic: 
for each team, go through each game of their schedule, and add the distance to go from the HOME TEAM  directly to the next game
ex. schedule is like Raptors -> Celtics (home), Raptors -> Warriors (home), Cavs -> Raptors (home)
distance = teams_distances(raptors, celtics) + teams_distances(celtics, warriors) + teams_distances(warriors, raptors)
"""
def calculate_total_distance(schedule):
    total_dist = 0
    team_schedule = create_team_schedule(schedule)

    for team in team_schedule:
        s = team_schedule[team]
        prev = team
        for j in range(len(s)):
            total_dist += teams_distances[(prev, s[j][2])]
            prev = s[j][2]
    return total_dist

"""
Calculates total number of back to back games played by all teams in given schedule
"""
def calculate_b2b_games(schedule):
    num_b2b = 0
    team_schedule = create_team_schedule(schedule)

    # calculate number of b2b games for each team, count, and return count
    for team in team_schedule:
        s = team_schedule[team]
        for j in range(1,len(s)):
            curr_game = s[j]
            prev_game = s[j-1]
            b2b = (team in curr_game) and (team in prev_game)
            delta = curr_game[0] - prev_game[0]
            if ((delta.days == 1 ) and b2b):
                num_b2b += 1
    return num_b2b

"""
Checks if given schedule complies with constraints
"""
def is_constraint_compliant(schedule):
    """
    constraints:
    1: 4 games against the other 4 division opponents (4×4=16 games)
    2: 4 games* against 6 (out-of-division) conference opponents (4×6=24 games)
    3: 3 games against the remaining 4 conference teams (3×4=12 games)
    4: 2 games against teams in the opposing conference (2×15=30 games)
    """
    team_schedule = create_team_schedule(schedule)

    # structure of count map is {(team1, team2) : num_games_played}
    count_map = {}
    for i in range(len(teams)):
        team1 = teams[i]
        for j in range(len(teams)):
            team2 = teams[j]
            count_map[(team1, team2)] = 0

    #populate count_map
    for team in teams_confs_divs:
        schedule = team_schedule[team]
        for game in schedule:
            if game[1] == team:
                opp = game[2]
                count_map[(team, opp)] += 1 
            elif game[2] == team:
                opp = game[1]
                count_map[(team, opp)] += 1

    # validate counts 
    for team in teams_confs_divs:
        conf = teams_confs_divs[team][0]
        div = teams_confs_divs[team][1]
        conf_teams = confs[conf]
        div_teams = divs[div]
        out_of_div_conf_teams = list(set(conf_teams) ^ set(div_teams)) # https://stackoverflow.com/a/40185809
        if (conf == "Eastern"):
            opp_conf_teams = confs["Western"]
        if (conf == "Western"):
            opp_conf_teams = confs["Eastern"]
        # constraint 1 check 
        for opp_team in div_teams:
            if team != opp_team:
                if count_map[(team, opp_team)] != 4:
                    return False
        # constraint 2 and 3 check 
        num_4_games = 0
        num_3_games = 0 
        for opp_team in out_of_div_conf_teams:
            if team != opp_team:
                if count_map[(team, opp_team)] == 4:
                    num_4_games += 1
                elif count_map[(team, opp_team)] == 3:
                    num_3_games += 1 
                else:
                    return False
        # constraint 4 check 
        for opp_team in opp_conf_teams:
            if count_map[(team, opp_team)] != 2:
                return False
    return True

"""
Returns objective value of given schedule
"""
def get_objective_value(schedule):
    w1 = 0.5
    w2 = 0.5
    return (w1 * calculate_total_distance(schedule)) + (w2 * calculate_b2b_games(schedule))

"""
Finds neighboring schedule
"""
def get_neighbor_schedule(schedule):
    neighbor = copy.deepcopy(schedule)
    # random swapping of games in schedule
    for i in range(5):
        r1 = random.randint(0, len(schedule)-1)
        r2 = random.randint(0, len(schedule)-1)
        temp_visitor = neighbor[r1][1]
        temp_home = neighbor[r1][2]
        neighbor[r1][1], neighbor[r1][2] = neighbor[r2][1], neighbor[r2][2]
        neighbor[r2][1], neighbor[r2][2] = temp_visitor, temp_home
    return neighbor
    
"""
SETUP (applicable for any schedule)
"""

t = pd.read_csv('teams.csv')
c = 0 
teams = []
arenas = []
for i in range(len(t)):
    team = t.iloc[i]
    teams.append(team["CITY"] + " " + team["NICKNAME"])
    arenas.append(team["ARENA"])
    c+= 1

#creating map of each team and their division and conference
# structure is {team_name : [conference, division]}
teams_confs_divs = {}
confs_divs = pd.read_csv('divisions_conferences.csv') 

# creating maps for all teams in each conference and division
# structure is confs = {conference : [teams in conf]} and divs = {division : [teams in division]}
confs = {}
confs["Eastern"] = []
confs["Western"] = []
divs = {}
divs["Atlantic"] = []
divs["Central"] = []
divs["Southeast"] = []
divs["Northwest"] = []
divs["Pacific"] = []
divs["Southwest"] = []

for i in range(len(confs_divs)):
    curr = confs_divs.iloc[i]
    team = curr["Team"]
    conf = curr["Conference"]
    div = curr["Division"]
    teams_confs_divs[team] = [conf,div]
    confs[conf].append(team)
    divs[div].append(team)

# creating map of arena with latitude longitude information
# structure is array of arrays like [team name, latitude, longitude]
teams_coords = []
coords = pd.read_csv('coords.csv') 

for i in range(len(coords)):
    coord = coords.iloc[i]
    curr = []
    curr.append(coord["Team"])
    curr.append(coord["Latitude"])
    curr.append(coord["Longitude"])
    teams_coords.append(curr)

#print(teams_coords)

# mapping distances between all arenas
teams_distances = {}

for i in range(len(teams_coords)):
    for j in range(len(teams_coords)):
        team1 = teams_coords[i][0]
        team2 = teams_coords[j][0]
        lat1 = teams_coords[i][1]
        lon1 = teams_coords[j][1]
        lat2 = teams_coords[i][2]
        lon2 = teams_coords[j][2]
        teams_distances[(team1, team2)] = haversine(lat1, lon1, lat2, lon2)

"""
SCHEDULE(s) 
building structure for 2020 nba season schedule
"""

#SCHEDULE 1 

games = pd.read_csv('games.csv')

# structure will be array of [datetime obj of game, visiting team, home team]
schedule = []

for i in range(len(games)):
    game = games.iloc[i]
    curr = []
    date = parser.parse(game["Date"])
    visitor = game["Visitor/Neutral"]
    home = game["Home/Neutral"]
    curr.append(date)
    curr.append(visitor)
    curr.append(home)
    schedule.append(curr)

print(get_objective_value(schedule))

# SCHEDULE 2 
# games2 = pd.read_csv('games2.csv')

# structure will be array of [datetime obj of game, visiting team, home team]
# schedule2 = []

# for i in range(len(games2)):
#     game2 = games2.iloc[i]
#     curr = []
#     date = parser.parse(game2["Date"])
#     visitor = game2["Visitor/Neutral"]
#     home = game2["Home/Neutral"]
#     curr.append(date)
#     curr.append(visitor)
#     curr.append(home)
#     schedule2.append(curr)

# # swapping games in schedule
# for i in range(100):
#     r1 = random.randint(0, len(schedule)-1)
#     r2 = random.randint(0, len(schedule)-1)
#     temp_visitor = schedule2[r1][1]
#     temp_home = schedule2[r1][2]
#     schedule2[r1][1], schedule2[r1][2] = schedule2[r2][1], schedule2[r2][2]
#     schedule2[r2][1], schedule2[r2][2] = temp_visitor, temp_home

# print(get_objective_value(schedule2))

"""
Simulated Annealing solution
"""

k = 1
T = 100
num_iterations = 100
a = 0.9

temp = T*(a**k)
s0 = copy.deepcopy(schedule)
sk = copy.deepcopy(schedule)

initial_schedule_obj = get_objective_value(schedule)
print("Initial objective function value of schedule is: " + str(initial_schedule_obj))

while k <= num_iterations:
    sc = get_neighbor_schedule(sk)
    obj_s0 = get_objective_value(s0)
    obj_sc = get_objective_value(sc)
    obj_sk = get_objective_value(sk)

    if (obj_s0 < obj_sc and obj_sc < obj_sk):
        sk = sc
    elif (obj_sc < obj_s0):
        s0 = sc
        sk = sc
    elif (obj_sc > obj_sk):
        uk = random.uniform(0,1)
        prob = math.exp(-1*(obj_sc - obj_sk) / temp)
        if (uk <= prob):
            sk = sc
        # else sk remains the same 
    k += 1
    temp = T*(a**k)

final_schedule_obj = get_objective_value(sk)
print("Final objective function value of updated schedule is: " + str(final_schedule_obj))
