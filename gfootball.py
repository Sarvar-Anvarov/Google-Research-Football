from kaggle_environments.envs.football.helpers import *
import math
import random
import numpy as np
import heapq

def find_patterns(obs, player_x, player_y):
    """ find list of appropriate patterns in groups of memory patterns """
    for get_group in groups_of_memory_patterns:
        group = get_group(obs, player_x, player_y)
        if group["environment_fits"](obs, player_x, player_y):
            return group["get_memory_patterns"](obs, player_x, player_y)

        
def get_action_of_agent(obs, player_x, player_y):
    """ get action of appropriate pattern in agent's memory """
    memory_patterns = find_patterns(obs, player_x, player_y)
    # find appropriate pattern in list of memory patterns
    for get_pattern in memory_patterns:
        pattern = get_pattern(obs, player_x, player_y)
        if pattern["environment_fits"](obs, player_x, player_y):
            return pattern["get_action"](obs, player_x, player_y)

        
def get_distance(x1, y1, right_team):
    """ get two-dimensional Euclidean distance, considering y size of the field """
    return math.sqrt((x1 - right_team[0]) ** 2 + (y1 * 2.38 - right_team[1] * 2.38) ** 2)


def run_to_ball_bottom(obs, player_x, player_y):
    """ run to the ball if it is to the bottom from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom from player's position
        if (obs["ball"][1] > player_y and
                abs(obs["ball"][0] - player_x) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Bottom
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_bottom_left(obs, player_x, player_y):
    """ run to the ball if it is to the bottom left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom left from player's position
        if (obs["ball"][0] < player_x and
                obs["ball"][1] > player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.BottomLeft
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_bottom_right(obs, player_x, player_y):
    """ run to the ball if it is to the bottom right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the bottom right from player's position
        if (obs["ball"][0] > player_x and
                obs["ball"][1] > player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.BottomRight
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_left(obs, player_x, player_y):
    """ run to the ball if it is to the left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the left from player's position
        if (obs["ball"][0] < player_x and
                abs(obs["ball"][1] - player_y) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Left
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_right(obs, player_x, player_y):
    """ run to the ball if it is to the right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the right from player's position
        if (obs["ball"][0] > player_x and
                abs(obs["ball"][1] - player_y) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Right
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_top(obs, player_x, player_y):
    """ run to the ball if it is to the top from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top from player's position
        if (obs["ball"][1] < player_y and
                abs(obs["ball"][0] - player_x) < 0.01):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Top
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_top_left(obs, player_x, player_y):
    """ run to the ball if it is to the top left from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top left from player's position
        if (obs["ball"][0] < player_x and
                obs["ball"][1] < player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.TopLeft
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def run_to_ball_top_right(obs, player_x, player_y):
    """ run to the ball if it is to the top right from player's position """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # ball is to the top right from player's position
        if (obs["ball"][0] > player_x and
                obs["ball"][1] < player_y):
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.TopRight
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def idle(obs, player_x, player_y):
    """ do nothing, release all sticky actions """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        return True
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        return Action.Idle
    
    return {"environment_fits": environment_fits, "get_action": get_action}
 
    
def start_sprinting(obs, player_x, player_y):
    """ make sure player is sprinting """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        if Action.Sprint not in obs["sticky_actions"]:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Dribble in obs['sticky_actions']:
            return Action.ReleaseDribble
        return Action.Sprint
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def corner(obs, player_x, player_y):
    """ perform a shot in corner game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is corner game mode
        if obs['game_mode'] == GameMode.Corner:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y > 0:
            if Action.TopRight not in obs["sticky_actions"]:
                return Action.TopRight
        else:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        return Action.HighPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def free_kick(obs, player_x, player_y):
    """ perform a high pass or a shot in free kick game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is free kick game mode
        if obs['game_mode'] == GameMode.FreeKick:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        # shot if player close to goal
        if player_x > 0.5:
            if player_y > 0:
                if Action.TopRight not in obs["sticky_actions"]:
                    return Action.TopRight
            else:
                if Action.BottomRight not in obs["sticky_actions"]:
                    return Action.BottomRight
            return Action.Shot
        # high pass if player far from goal
        else:
            if player_y > 0:
                if Action.BottomRight not in obs["sticky_actions"]:
                    return Action.BottomRight
            else:
                if Action.TopRight not in obs['sticky_actions']:
                    return Action.TopRight
            return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def goal_kick(obs, player_x, player_y):
    """ perform a short pass in goal kick game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is goal kick game mode
        if obs['game_mode'] == GameMode.GoalKick:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.BottomRight not in obs["sticky_actions"]:
            return Action.BottomRight
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def kick_off(obs, player_x, player_y):
    """ perform a short pass in kick off game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is kick off game mode
        if obs['game_mode'] == GameMode.KickOff:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y > 0:
            if Action.Top not in obs["sticky_actions"]:
                return Action.Top
        else:
            if Action.Bottom not in obs["sticky_actions"]:
                return Action.Bottom
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def penalty(obs, player_x, player_y):
    """ perform a shot in penalty game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is penalty game mode
        if obs['game_mode'] == GameMode.Penalty:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if (random.random() < 0.5 and
                Action.TopRight not in obs["sticky_actions"] and
                Action.BottomRight not in obs["sticky_actions"]):
            return Action.TopRight
        else:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def throw_in(obs, player_x, player_y):
    """ perform a short pass in throw in game mode """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # it is throw in game mode
        if obs['game_mode'] == GameMode.ThrowIn:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if Action.Right not in obs["sticky_actions"]:
            return Action.Right
        return Action.ShortPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def defence_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for environments in which opponent's team has the ball """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player don't have the ball
        if obs["ball_owned_team"] != 0:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        # shift ball position
        obs["ball"][0] += obs["ball_direction"][0] * 7
        obs["ball"][1] += obs["ball_direction"][1] * 3
        # if opponent has the ball and is far from y axis center
        if abs(obs["ball"][1]) > 0.07 and obs["ball_owned_team"] == 1:
            obs["ball"][0] -= 0.01
            if obs["ball"][1] > 0:
                obs["ball"][1] -= 0.01
            else:
                obs["ball"][1] += 0.01
            
        memory_patterns = [
            start_sprinting,
            run_to_ball_right,
            run_to_ball_left,
            run_to_ball_bottom,
            run_to_ball_top,
            run_to_ball_top_right,
            run_to_ball_top_left,
            run_to_ball_bottom_right,
            run_to_ball_bottom_left,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

def goalkeeper_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for goalkeeper """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player is a goalkeeper have the ball
        if (obs["ball_owned_player"] == obs["active"] and
                obs["ball_owned_team"] == 0 and
                obs["ball_owned_player"] == 0):
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            long_pass_forward,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}


def offence_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for environments in which player's team has the ball """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # player have the ball
        if obs["ball_owned_player"] == obs["active"] and obs["ball_owned_team"] == 0:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            close_to_goalkeeper_shot,
            spot_shot,
            cross,
            long_pass_forward,
            keep_the_ball,
        idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}


def other_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for all other environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        return True
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}

def special_game_modes_memory_patterns(obs, player_x, player_y):
    """ group of memory patterns for special game mode environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # if game mode is not normal
        if obs['game_mode'] != GameMode.Normal:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            corner,
            free_kick,
            goal_kick,
            kick_off,
            penalty,
            throw_in,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}


def special_spot_shot(obs, player_x, player_y):
    """ group of memory patterns for special game mode environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # if game mode is not normal
        if player_x > 0.8 and abs(player_y) < 0.21:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            shot,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}


def own_goal(obs, player_x, player_y):
    """ group of memory patterns for special game mode environments """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # if game mode is not normal
        if player_x < -0.9:
            return True
        return False
        
    def get_memory_patterns(obs, player_x, player_y):
        """ get list of memory patterns """
        memory_patterns = [
            own_goal,
            idle
        ]
        return memory_patterns
        
    return {"environment_fits": environment_fits, "get_memory_patterns": get_memory_patterns}


# list of groups of memory patterns
groups_of_memory_patterns = [
    special_spot_shot,
    special_game_modes_memory_patterns,
    own_goal,
    goalkeeper_memory_patterns,
    offence_memory_patterns,
    defence_memory_patterns,
    other_memory_patterns
]


def keep_the_ball(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        return True
    
    def get_action(obs, player_x, player_y):
        right_team, left_team = obs['right_team'], obs['left_team']
        dist = [get_distance(player_x, player_y, i) for i in right_team]
        closest = right_team[np.argmin(dist)]
        near = [i for i in right_team if (i[0] < player_x + 0.2) and (i[0] > player_x) and (i[1] > player_y - 0.05)
               and (i[1] < player_y + 0.05)] 
        back = [i for i in right_team if (i[0] > player_x)]
        bottom_right = [i for i in left_team if (i[0] > player_x - 0.05) and (i[0] < player_x + 0.2) and (i[1] < player_y + 0.2) and 
                       (i[1] > player_y)]
        top_right = [i for i in left_team if (i[0] > player_x - 0.05) and (i[0] < player_x + 0.2) and (i[1] > player_y - 0.2) and 
                       (i[1] < player_y)]
        bottom_left = [i for i in left_team if (i[0] < player_x) and (i[0] > player_x - 0.2) and (i[1] < player_y + 0.2) and 
                       (i[1] > player_y)]
        top_left = [i for i in left_team if (i[0] < player_x) and (i[0] > player_x - 0.2) and (i[1] > player_y - 0.2) and 
                       (i[1] < player_y)]
        
    
        if len(near) == 0:
            return Action.Right
        
        if player_y > 0:
            if player_y > 0.35:
                return Action.Right
            if len(bottom_right) > 0:
                if Action.BottomRight not in obs['sticky_actions']:
                    return Action.BottomRight
                return Action.ShortPass
            return Action.BottomRight
        
        if player_y < 0:
            if player_y < -0.35:
                return Action.Right
            if len(top_right) > 0:
                if Action.TopRight not in obs['sticky_actions']:
                    return Action.TopRight
                return Action.ShortPass
            return Action.TopRight
            
    return {'environment_fits': environment_fits, 'get_action': get_action}


def spot_shot(obs, player_x, player_y):
    """ shot if close to the goalkeeper """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        # shoot if in spotted location
        if player_x > 0.75 and abs(player_y) < 0.21:
            return True
        return False

    
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y >= 0:
            if Action.TopRight not in obs["sticky_actions"]:
                return Action.TopRight
        else:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        return Action.Shot

    return {"environment_fits": environment_fits, "get_action": get_action}


def cross(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        if player_x > 0.7 and abs(player_y) > 0.21:
            return True
        return False
    
    def get_action(obs, player_x, player_y):
        
        if player_x > 0.88:
            if player_y > 0:
                if Action.Top not in obs['sticky_actions']:
                    return Action.Top
            else:
                if Action.Bottom not in obs['sticky_actions']:
                    return Action.Bottom
            return Action.HighPass
        
        if player_x > 0.9:
            if (Action.Right in obs['sticky_actions'] or 
                Action.TopRight in obs['sticky_actions'] or 
                Action.BottomRight in obs['sticky_actions']):
                return Action.ReleaseDirection
            if Action.Right not in obs['sticky_actions']:
                if player_y > 0:
                    if Action.Top not in obs['sticky_actions']:
                        return Action.Top
                if player_y < 0:
                    if Action.Bottom not in obs['sticky_actions']:
                        return Action.Bottom
        return Action.HighPass
                
    return {"environment_fits": environment_fits, "get_action": get_action}


def close_to_goalkeeper_shot(obs, player_x, player_y):
    """ shot if close to the goalkeeper """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        goalkeeper_x = obs["right_team"][0][0] + obs["right_team_direction"][0][0] * 13
        goalkeeper_y = obs["right_team"][0][1] + obs["right_team_direction"][0][1] * 13
        goalkeeper = [goalkeeper_x,goalkeeper_y]
        
        if get_distance(player_x, player_y, goalkeeper) < 0.25:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        if player_y >= 0:
            if Action.TopRight not in obs["sticky_actions"]:
                return Action.TopRight
        else:
            if Action.BottomRight not in obs["sticky_actions"]:
                return Action.BottomRight
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def long_pass_forward(obs, player_x, player_y):
    """ perform a high pass, if far from opponent's goal """
    def environment_fits(obs, player_x, player_y):
        """ environment fits constraints """
        right_team = obs["right_team"][1:]
        # player have the ball and is far from opponent's goal
        if player_x < -0.4:
            return True
        return False
        
    def get_action(obs, player_x, player_y):
        """ get action of this memory pattern """
        right_team, left_team = obs['right_team'], obs['left_team']
        dist = [get_distance(player_x, player_y, i) for i in right_team]
        closest = right_team[np.argmin(dist)]
        
        
        if abs(player_y) > 0.22:
            if Action.Right not in obs["sticky_actions"]:
                return Action.Right
            return Action.HighPass
        
        if np.min(dist) > 0.4:
            if player_y > 0:
                return Action.Bottom
            else:
                return Action.Top
            
        if np.min(dist) < 0.4 and np.min(dist) > 0.2:
            if player_y < 0:
                return Action.TopRight
            else:
                return Action.BottomRight
            
        if np.min(dist) < 0.2:
            if Action.Right not in obs['sticky_actions']:
                return Action.Right
            return Action.HighPass
    
    return {"environment_fits": environment_fits, "get_action": get_action}

def shot(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        return True
    
    def get_action(obs, player_x, player_y):
        if player_y > 0:
            if Action.TopRight not in obs['sticky_actions']:
                return Action.TopRight
        else:
            if Action.BottomRight not in obs['sticky_actions']:
                return Action.BottomRight
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}


def own_goal(obs, player_x, player_y):
    def environment_fits(obs, player_x, player_y):
        if player_x < -0.92:
            return True
        return False
    
    def get_action(obs, player_x, player_y):
        return Action.Shot
    
    return {"environment_fits": environment_fits, "get_action": get_action}


# @human_readable_agent wrapper modifies raw observations 
# provided by the environment:
# https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#raw-observations
# into a form easier to work with by humans.
# Following modifications are applied:
# - Action, PlayerRole and GameMode enums are introduced.
# - 'sticky_actions' are turned into a set of active actions (Action enum)
#    see usage example below.
# - 'game_mode' is turned into GameMode enum.
# - 'designated' field is removed, as it always equals to 'active'
#    when a single player is controlled on the team.
# - 'left_team_roles'/'right_team_roles' are turned into PlayerRole enums.
# - Action enum is to be returned by the agent function.
@human_readable_agent
def agent(obs):
    """ Ole ole ole ole """
    # dictionary for Memory Patterns data
    obs["memory_patterns"] = {}
    # We always control left team (observations and actions
    # are mirrored appropriately by the environment).
    controlled_player_pos = obs["left_team"][obs["active"]]
    # get action of appropriate pattern in agent's memory
    action = get_action_of_agent(obs, controlled_player_pos[0], controlled_player_pos[1])
    # return action
    return action
