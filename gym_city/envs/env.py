from gym import core, spaces
import gym
from gym.utils import seeding
from .tilemap import zoneFromInt
from collections import OrderedDict
import numpy as np
import math

import sys

if sys.version_info[0] >= 3:
    print(sys.version_info[0])
    import gi

    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk as gtk
    from .tilemap import TileMap
    from .corecontrol import MicropolisControl
else:
    import gtk
    from tilemap import TileMap
    from corecontrol import MicropolisControl
import time
import torch


class MicropolisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, MAP_X=20, MAP_Y=20, PADDING=0):
        super(MicropolisEnv, self).__init__()
        self.SHOW_GUI = False
        self.start_time = time.time()
        self.num_episode = 0
        self.player_step = False

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        np.random.seed(seed)
        return [seed1, seed2]

    def setMapSize(self, size, **kwargs):
        '''Do most of the actual initialization.
        '''
        self.pre_gui(size, **kwargs)
        # TODO: this better
        if hasattr(self, 'micro'):
            self.micro.reset_params(size)
        else:
            self.micro = MicropolisControl(self, self.MAP_X, self.MAP_Y, self.PADDING,
                                           rank=self.rank, power_puzzle=self.power_puzzle, gui=self.render_gui)
        self.post_gui()

    def pre_gui(self, size, max_step=None, rank=0, print_map=False,
                PADDING=0, static_builds=True, parallel_gui=False,
                render_gui=False, empty_start=True, simple_reward=False,
                power_puzzle=True, record=False, traffic_only=False, random_builds=False, poet=False, **kwargs):
        self.PADDING = PADDING
        self.rank = rank
        self.render_gui = render_gui
        self.random_builds = random_builds

        if max_step is None:
            max_step = size * size

        self.max_step = max_step

        self.empty_start = empty_start

        self.power_puzzle = power_puzzle

        if type(size) == int:
            self.MAP_X = size
            self.MAP_Y = size
        else:
            self.MAP_X = size[0]
            self.MAP_Y = size[1]

        self.obs_width = self.MAP_X + PADDING * 2
        self.static_builds = True

    def post_gui(self):
        self.win1 = self.micro.win1
        self.micro.SHOW_GUI = self.SHOW_GUI
        self.num_step = 0
        self.minFunds = 0
        self.initFunds = self.micro.init_funds
        self.num_tools = self.micro.num_tools
        self.num_zones = self.micro.num_zones
        # res, com, ind pop, demand
        self.num_scalars = 6
        self.num_density_maps = 3
        num_user_features = 1  # static builds
        # traffic, power, density
        print('num map features: {}'.format(self.micro.map.num_features))
        self.num_obs_channels = 34
        self.action_space = spaces.Discrete(self.num_tools * self.MAP_X * self.MAP_Y)
        self.last_state = None
        # self.metadata = {'runtime.vectorized': True}
        # Define the observation space as a flattened 1D array
        low_obs = np.full((self.num_obs_channels * self.MAP_X * self.MAP_Y,), fill_value=-1)
        high_obs = np.full((self.num_obs_channels * self.MAP_X * self.MAP_Y,), fill_value=1)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float)
        self.state = None
        self.intsToActions = {}
        self.actionsToInts = np.zeros((self.num_tools, self.MAP_X, self.MAP_Y))
        self.mapIntsToActions()
        self.last_pop = 0
        self.last_n_zones = 0
        self.last_num_roads = 0
        #       self.past_actions = np.full((self.num_tools, self.MAP_X, self.MAP_Y), False)
        self.auto_reset = True
        self.mayor_rating = 50
        self.last_mayor_rating = self.mayor_rating
        self.last_priority_road_net_size = 0
        if self.render_gui and self.rank == 0:
            self.render()

    def mapIntsToActionsChunk(self):
        ''' Unrolls the action vector into spatial chunks (does this matter empirically?).'''
        w0 = 20
        w1 = 10
        i = 0
        for j0 in range(self.MAP_X // w0):
            for k0 in range(self.MAP_Y // w0):
                for j1 in range(w0 // w1):
                    for k1 in range(w0 // w1):
                        for z in range(self.num_tools):
                            for x in range(j0 * w0 + j1 * w1,
                                           j0 * w0 + (j1 + 1) * w1):
                                for y in range(k0 * w0 + k1 * w1,
                                               k0 * w0 + (k1 + 1) * w1):
                                    self.intsToActions[i] = [z, x, y]
                                    i += 1

    def mapIntsToActions(self):
        ''' Unrolls the action vector in the same order as the pytorch model
        on its forward pass.'''
        chunk_width = 1
        i = 0
        for z in range(self.num_tools):
            for x in range(self.MAP_X):
                for y in range(self.MAP_Y):
                    self.intsToActions[i] = [z, x, y]
                    self.actionsToInts[z, x, y] = i
                    i += 1
        print('len of intsToActions: {}\n num tools: {}'.format(len(self.intsToActions), self.num_tools))

    def close(self):
        self.micro.close()

    def powerPuzzle(self):
        ''' Set up one plant, one res. If we restrict the agent to building power lines, we can test its ability
        to make long-range associations. '''
        for i in range(5):
            self.micro.doBotTool(np.random.randint(0, self.micro.MAP_X),
                                 np.random.randint(0, self.micro.MAP_Y), 'Residential', static_build=True)
        while self.micro.map.num_plants == 0:
            self.micro.doBotTool(np.random.randint(0, self.micro.MAP_X),
                                 np.random.randint(0, self.micro.MAP_Y),
                                 'NuclearPowerPlant', static_build=True)

    def reset(self, prebuild=True):
        self.micro.clearMap()
        if not self.empty_start:
            self.micro.newMap()
        self.num_step = 0
        if self.power_puzzle:
            self.powerPuzzle()
        self.micro.simTick()
        self.micro.setFunds(self.micro.init_funds)
        # curr_funds = self.micro.getFunds()
        self.curr_pop = 0
        self.curr_reward = self.getReward()
        self.state = self.getState()
        self.last_pop = 0
        self.last_n_zones = 0
        self.micro.num_roads = 0
        self.last_num_roads = 0
        self.last_networks = None
        self.num_episode += 1

        return self.state

    def getState(self):
        res_pop, com_pop, ind_pop = self.micro.getResPop(), self.micro.getComPop(), self.micro.getIndPop()
        resDemand, comDemand, indDemand = self.micro.engine.getDemands()
        scalars = [res_pop, com_pop, ind_pop, resDemand, comDemand, indDemand]
        return self.observation(scalars)

    def get_building_map(self, text=True):
        building_map = []
        if text:
            for x in range(self.MAP_X):
                row_buildings = []
                for y in range(self.MAP_Y):
                    tile = (self.micro.getTile(x, y))
                    tile = zoneFromInt(tile)
                    row_buildings.append(tile)
                building_map.append(row_buildings)
        else:
            for x in range(self.MAP_X):
                row_buildings = []
                for y in range(self.MAP_Y):
                    tile = (self.micro.getTile(x, y))
                    row_buildings.append(tile)
                building_map.append(row_buildings)
        return building_map

    def observation(self, scalars):
        building_map = self.get_building_map(text=False)
        simple_state = self.micro.map.getMapState()
        density_maps = self.micro.getDensityMaps()
        scalar_layers = np.zeros((len(scalars), self.MAP_X, self.MAP_Y))
        for si in range(len(scalars)):
            fill_val = scalars[si]
            if not type(fill_val) == str and -1.0 <= fill_val <= 1.0:
                scalar_layers[si].fill(fill_val)
            else:
                scalar_layers[si].fill(0)
        state = np.concatenate((simple_state, scalar_layers), 0)

        if self.static_builds:
            state = np.concatenate((state, density_maps, self.micro.map.static_builds), 0)

        state = np.array(state).flatten()

        # unique_values, counts = np.unique(state, return_counts=True)
        return state

    def getPop(self):
        self.resPop, self.comPop, self.indPop = self.micro.getResPop(), \
            self.micro.getComPop(), \
            self.micro.getIndPop()

        curr_pop = self.resPop + \
                   self.comPop + \
                   self.indPop

        return curr_pop

    def getReward(self, action=None):
        reward = self.micro.getPoweredZoneCount() + self.getPop()
        return reward

    def step(self, a, static_build=False):
        if self.player_step:
            a = self.player_step
            self.player_step = False
        if isinstance(a, np.ndarray):
            a = self.intsToActions[a[0]]
        else:
            a = self.intsToActions[a]
        self.micro.takeAction(a, static_build)
        return self.postact(a)

    def postact(self, action=None):
        # never let the agent go broke, for now
        self.micro.setFunds(self.micro.init_funds)

        # TODO: BROKEN!
        self.micro.simTick()

        self.state = self.getState()

        self.curr_pop = self.getPop()

        reward = self.getReward(action=action)

        self.curr_funds = curr_funds = self.micro.getFunds()

        bankrupt = curr_funds < self.minFunds

        if False:  # self.power_puzzle:
            terminal = (self.micro.getPoweredZoneCount() == self.micro.getTotalZonePop() + 1
                        or self.num_step >= self.max_step) and self.auto_reset
        else:
            terminal = (bankrupt or self.num_step >= self.max_step) and \
                       self.auto_reset

        if self.render_gui:
            self.micro.render()
        infos = {}
        # Get the next player-build ready, if there is one in the queue
        if self.micro.player_builds:
            b = self.micro.player_builds[0]
            a = self.actionsToInts[b]
            infos['player_move'] = int(a)
            self.micro.player_builds = self.micro.player_builds[1:]
            self.player_step = a

        self.num_step += 1
        return self.state, reward, terminal, infos

    def render(self, mode='human'):
        self.micro.render()
