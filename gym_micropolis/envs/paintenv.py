from gym import core, spaces
from gym.utils import seeding
import numpy as np
import math

import sys
if sys.version_info[0] >= 3:
    import gi
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk as gtk
    from .tilemap import TileMap
    from .paintcontrol import MicropolisPaintControl
else:
    import gtk
    from tilemap import TileMap
    from paintcontrol import MicropolisPaintControl
import time
import torch
from .env import MicropolisEnv

class MicropolisPaintEnv(MicropolisEnv):

    def __init__(self, MAP_X=20, MAP_Y=20, PADDING=0):
        super(MicropolisPaintEnv, self).__init__(MAP_X, MAP_Y, PADDING)

    def setMapSize(self, size, **kw_args):
        super().setMapSize(size, **kw_args)
        ac_low = np.zeros((self.MAP_X, self.MAP_Y))
        ac_high = np.zeros((self.MAP_X, self.MAP_Y))
        ac_high.fill(self.num_tools - 1)
        self.action_space = spaces.Box(low=ac_low, high=ac_high, dtype=int)

        self.micro = MicropolisPaintControl(self, MAP_W=self.MAP_X,
                MAP_H=self.MAP_Y, PADDING=self.PADDING,
                rank=self.rank, gui=self.render_gui,
                power_puzzle=self.power_puzzle)

    def step(self, a, static_build=True):
        '''
         - a: has shape (w, h)
        '''
        if self.player_step:
            if self.player_step == a:
                static_build=False
            self.player_step = None
        self.micro.takeAction(a, static_build)
        state, dummy_rew, terminal, infos = self.poststep()
        reward = self.getReward()
        return state, reward, terminal, infos


