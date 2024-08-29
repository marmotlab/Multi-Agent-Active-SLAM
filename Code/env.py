import random as rd
import copy
import time

#import matplotlib.pyplot as plt #not used
import numpy as np
from matplotlib.colors import hsv_to_rgb

from agent import Agent

from param import *
from tools import *

if MetaParameters.RENDER:
    from gym.envs.classic_control import rendering #works with gym==0.21.0 installed
    from functions.render import *


class Map:
    def __init__(self, imposed_map):
        #create map
        if imposed_map is None:
            self.height = EnvParameters.MAP_HEIGHT
            self.width = EnvParameters.MAP_WIDTH
            self.occupancy_rate = EnvParameters.OCC_RATE
            self.occupancy_grid = self.generate_map()
        else:
            self.occupancy_grid = imposed_map
            self.height = np.size(self.occupancy_grid,0)
            self.width = np.size(self.occupancy_grid,1)
            self.occupancy_rate = None

    def generate_map(self):
        occ_grid = np.random.choice([0,1], [self.height, self.width], p=[1-self.occupancy_rate, self.occupancy_rate])
        return occ_grid

    def print_map(self):
        print("Here is the map : ")
        print(self.occupancy_grid)

    def is_out(self, pos):
        return pos[0] < 0 or self.height <= pos[0] or pos[1] < 0 or self.width <= pos[1]

    def get_square(self, pos):
        #return self.occupancy_grid[self.height - pos[1] - 1, pos[0]]
        if self.is_out(pos):
            return -1
        else:
            return self.occupancy_grid[pos[0], pos[1]]
    
    def set_rd_point(self):
        return (rd.randrange(self.height), rd.randrange(self.width))

    def set_rd_free_point(self):
        while True:
            rd_point = self.set_rd_point()
            if self.get_square(rd_point) == 0:
                break
        return (rd_point)
    
    def display_init(self, disp = False):
        print("A new map has been created!")
        if disp :
            print("Map dimensions :", self.height, " height,", self.width, " width")
            print("Occupancy rate :", self.occupancy_rate)
            self.print_map()

    


    


class MA_SLAM_Env:
    def __init__(self, n_agents, n_steps, imposed_map = None, init_pos = None):
        print("Hello world! Welcome in the new Environment.")
        #init world
        self.n_agents = n_agents
        
        #init map
        self.init_map = Map(imposed_map)
        self.map = copy.deepcopy(self.init_map)

        #init team
        self.n_agents = n_agents
        self.team = {}
        self.init_pos = self.get_init_poses(init_pos)
        for i_agent in range(self.n_agents):
            id = i_agent+1
            color = hsv_to_rgb(np.array([(id-1)/float(self.n_agents),1,1]))
            self.team[id] = Agent(id, color, self.init_pos[id], self.map)

        #init episode
        self.n_steps = n_steps
        self.time_step = 0
        self.running = False
        self.done = False

        #init render variables
        self.screen_dim = RenderingParameters.SCREEN_DIM
        self.world_viewer = None
        self.episode_world_frames = []

    def get_init_poses(self, init_poses):
        if init_poses is not None:
            pot_init_poses = {i_agent+1 : init_poses[i_agent] for i_agent in range(self.n_agents)}
        else:
            pot_init_poses = {i_agent+1 : (rd.randrange(0, self.map.height), rd.randrange(0, self.map.width)) for i_agent in range(self.n_agents)}
        
        #check if the init poses is compatible with the map, update them if not
        for k in pot_init_poses:
            i_agent = k
            if self.init_map.get_square(pot_init_poses[i_agent]) == 1: #if the init pos is not free
                k=0
                while k<10:
                    k+=1
                    #change init pose
                    pot_init_poses[i_agent] = (rd.randrange(0, self.map.height), rd.randrange(0, self.map.width))
                    #get out of the loop if the square is free
                    if self.init_map.get_square(pot_init_poses[i_agent]) == 0:
                        break
        return pot_init_poses

    def is_task_done(self):
        for _id, agent in self.team.items():
            if agent.done != True:
                return False
        return True

    def reset(self):
        #reset episode
        self.time_step = 0
        self.running = False
        self.done = False
        
        #reset map
        self.map = self.init_map

        #reset team
        for _id, agent in self.team.items():
            agent.reset_agent()

        #reset render variables
        self.world_viewer = None
        self.episode_world_frames = []
        
        print("Environment reset (episode and world reset)")

    def update_world(self, new_map = None, changes = {}, team_changes = {}, disp = False):
        if disp : print("Updating the world ...")

        if new_map != None or changes != {} or team_changes != {}:

            #updating map
            if new_map != None:
                self.map = new_map
            elif changes != {}:
                updated_map = self.map
                for pos in changes:
                    updated_map[pos[0], pos[1]] = changes[pos] 
                self.map = updated_map
            
            if team_changes != {}:
                updated_team = self.team
                for _id, agent in team_changes.items():
                    pass #to be completed
                self.team = updated_team

    def display_init(self):
        #display init
        self.map.display_init(True)
        for _id, agent in self.team.items():
            agent.display_init()

    def display_world(self):
        print("\n----------World Display----------")
        self.map.print_map()
        for _id, agent in self.team.items():
            agent.move_base.display_move_base()

    def render_world(self, mode = 'human'):
        #variables
        screen_dim = self.screen_dim
        width = self.map.width
        height = self.map.height
        square_size = min(screen_dim[0]/width, screen_dim[1]/height)
        colors = {a+1:hsv_to_rgb(np.array([a/float(self.n_agents),1,1])) for a in range(len(self.team))}

        if self.world_viewer is None:
            self.world_viewer = rendering.Viewer(screen_dim[0], screen_dim[1])

            #white screen background
            create_rectangle(self.world_viewer, 0, screen_dim[1], screen_dim[0], screen_dim[1], (1,1,1), permanent = True)
            
            #grey map background
            create_rectangle(self.world_viewer, 0, screen_dim[1], square_size*width, square_size*height,(.8,.8,.8), permanent = True)

            #drawing the map
            for i in range(height):
                start = 0
                end = 1
                scanning = False
                write = False
                for j in range(width):
                    pos = (i,j)
                    square = self.map.get_square(pos)
                    if(square == 1 and not scanning): #obstacle found
                        start = j
                        scanning = True
                    if scanning:
                        if square == 0 :
                            end = j
                            scanning = False
                            write = True
                        elif j == width-1:
                            end = j+1
                            scanning = False
                            write = True
                    if write:
                        x = start*square_size
                        y = screen_dim[1] - i*square_size
                        create_rectangle(self.world_viewer, x, y, square_size*(end-start), square_size, (0.2,0.2,0.2), permanent = True)
                        write = False
        
        #drawing agents' position
        for _id, agent in self.team.items():
            pos = agent.move_base.pos
            x = pos[1]*square_size
            y = screen_dim[1] - pos[0]*square_size
            color = colors[agent.id]
            create_rectangle(self.world_viewer, x, y, square_size, square_size, color)

    def render(self):
        print("Rendering the world ...")
        self.render_world(mode = 'rgb_array')
        self.episode_world_frames.append(self.world_viewer.render(return_rgb_array = True))

    def close_viewers(self):
        if self.world_viewer:
            self.world_viewer.close()
            self.world_viewer = None

        for _id, agent in self.team.items():
            agent.close_viewers()




if __name__ == '__main__':
    
    #init env
    theEnv = MA_SLAM_Env(EnvParameters.N_AGENTS, EnvParameters.N_STEPS)

    #meta variables
    display = MetaParameters.DISPLAY
    render = MetaParameters.RENDER
    record = MetaParameters.RECORD
    measure_time = MetaParameters.MEASURE_TIME

    if display: theEnv.display_init()
    if render: theEnv.render()
    
    #init running variables
    theEnv.running = True

    for _id, agent in theEnv.team.items():
        agent.running = True
    
    try:
        #running
        while theEnv.running and theEnv.time_step < theEnv.n_steps:
            print("\nstep :", theEnv.time_step)
            #theEnv.multi_agent_step()
            if render: theEnv.render()

        #after running
        if measure_time:
            record_exec_time_stats(theEnv)
        #if record:
            #record_episode(theEnv)
        if render:
            theEnv.close_world_viewer()
            theEnv.close_agents_viewers()

    except KeyboardInterrupt:
        print("CTRL-C pressed. Process interrupted!")
        if render:
            theEnv.close_world_viewer()
            theEnv.close_agents_viewers()