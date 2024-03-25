from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import CombinedExtractor
from gym import spaces
import torch as th
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import copy
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = th.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data)
    bias_init(module.bias.data)
    return module


def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / th.sqrt(weight.pow(2).sum(1, keepdim=True))


class FractalNet(NNBase):
    def __init__(self, num_inputs, feature_dim: int,
                 last_layer_dim_pi: int = 64,
                 last_layer_dim_vf: int = 64, recurrent=False, hidden_size=512,
                 map_width=16, n_conv_recs=2, n_recs=1,
                 intra_shr=False, inter_shr=False,
                 num_actions=19, rule='extend',
                 in_w=1, in_h=1, out_w=1, out_h=1, n_chan=64, prebuild=None,
                 val_kern=3):
        super(FractalNet, self).__init__(recurrent, hidden_size, hidden_size)
        self.map_width = map_width
        # self.bn = nn.BatchNorm2d(num_inputs)
        # We can stack multiple Fractal Blocks
        # self.block_chans = block_chans = [32, 32, 16]
        self.block_chans = block_chans = [n_chan]
        self.num_blocks = num_blocks = len(block_chans)
        self.conv_init_ = init_ = lambda m: init(m,
                                                 nn.init.dirac_,
                                                 lambda x: nn.init.constant_(x, 0.1),
                                                 nn.init.calculate_gain('relu'))
        for i in range(num_blocks):
            setattr(self, 'block_{}'.format(i),
                    FractalBlock(n_chan_in=block_chans[i - 1], n_chan=block_chans[i],
                                 num_inputs=num_inputs, intra_shr=intra_shr,
                                 inter_shr=inter_shr, recurrent=recurrent,
                                 n_recs=n_recs,
                                 num_actions=num_actions, rule=rule, base=self))
        # An assumption. Run drop path globally on all blocks of stack if applicable
        self.n_cols = self.block_0.n_cols

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        n_out_chan = block_chans[-1]
        self.critic_dwn = init_(nn.Conv2d(n_out_chan, n_out_chan, val_kern, 2, 1))
        init_ = lambda m: init(m,
                               nn.init.dirac_,
                               lambda x: nn.init.constant_(x, 0))
        self.critic_out = init_(nn.Conv2d(n_out_chan, 1, 3, 1, 1))
        self.actor_out = init_(nn.Conv2d(n_out_chan, num_actions, 3, 1, 1))
        self.active_column = None

    def auto_expand(self):
        self.block_0.auto_expand()  # assumption
        self.n_cols += 1

    def forward(self, x, rnn_hxs=None, masks=None):
        # x = self.bn(x)
        for i in range(self.num_blocks):
            block = getattr(self, 'block_{}'.format(i))
            x = F.relu(block(x, rnn_hxs, masks))
        actions = self.actor_out(x)
        values = x
        for i in range(int(math.log(self.map_width, 2))):
            values = F.relu(self.critic_dwn(values))
        values = self.critic_out(values)
        values = values.view(values.size(0), -1)
        return values, actions, rnn_hxs  # no recurrent states

    def set_drop_path(self):
        for i in range(self.num_blocks):
            getattr(self, 'block_{}'.format(i)).set_drop_path()

    def set_active_column(self, a):
        self.active_column = a
        for i in range(self.num_blocks):
            getattr(self, 'block_{}'.format(i)).set_active_column(a)


class FractalBlock(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512,
                 map_width=16, n_recs=5, intra_shr=False,
                 inter_shr=False, num_actions=19, rule='extend', n_chan=32,
                 n_chan_in=32, base=None):

        super(FractalBlock, self).__init__(
            recurrent, hidden_size, hidden_size)

        self.map_width = map_width
        self.n_chan = n_chan
        self.intracol_share = intra_shr  # share weights between layers in a col.
        self.intercol_share = inter_shr  # share weights between columns
        self.rule = rule  # which fractal expansion rule to use
        # each rec is a call to a subfractal constructor, 1 rec = single-layered body
        self.n_recs = n_recs
        print("Fractal Block: expansion type: {}, {} recursions".format(
            self.rule, self.n_recs))

        self.SKIPSQUEEZE = rule == 'wide1'  # actually we mean a fractal rule that grows linearly in max depth but exponentially in number of columns, rather than vice versa, with number of recursions #TODO: combine the two rules
        if self.rule == 'wide1':
            self.n_cols = 2 ** (self.n_recs - 1)
            print('{} cols'.format(self.n_cols))
        else:
            self.n_cols = self.n_recs
        self.COLUMNS = False  # if true, we do not construct the network recursively, but as a row of concurrent columns
        # if true, weights are shared between recursions
        self.local_drop = False
        # at each join, which columns are taken as input (local drop as described in Fractal Net paper)
        self.global_drop = False
        self.active_column = None
        self.batch_norm = False
        self.c_init_ = init_ = lambda m: init(m,
                                              nn.init.dirac_,
                                              lambda x: nn.init.constant_(x, 0.1),
                                              nn.init.calculate_gain('relu'))
        self.embed_chan = nn.Conv2d(num_inputs, n_chan, 1, 1, 0)
        # TODO: right now, we initialize these only as placeholders to successfully load older models, get rid of these ASAP
        if False and self.intracol_share:
            # how many columns with distinct sets of layers?
            if self.intercol_share:
                n_unique_cols = 1
            else:
                n_unique_cols = self.n_recs
            for i in range(n_unique_cols):
                if self.intracol_share:
                    n_unique_layers = 1
                else:
                    n_unique_layers = 3
                setattr(self, 'fixed_{}'.format(i), init_(nn.Conv2d(
                    self.n_chan, self.n_chan, 3, 1, 1)))
                if n_unique_cols == 1 or i > 0:

                    setattr(self, 'join_{}'.format(i), init_(nn.Conv2d(
                        self.n_chan * 2, self.n_chan, 3, 1, 1)))
                    if self.rule == 'wide1' or self.rule == 'extend_sqz':
                        setattr(self, 'dwn_{}'.format(i), init_(nn.Conv2d(
                            self.n_chan, self.n_chan, 2, 2, 0)))
                        setattr(self, 'up_{}'.format(i), init_(nn.ConvTranspose2d(
                            self.n_chan, self.n_chan, 2, 2, 0)))
        f_c = None
        if self.rule == 'wide1':
            subfractal = SkipFractal
        elif self.rule == 'extend':
            if self.rule == 'extend_sqz':
                subfractal = SubFractal_squeeze
            else:
                subfractal = SubFractal
        n_recs = self.n_recs
        for i in range(n_recs):
            f_c = subfractal(self, f_c, n_rec=i, n_chan=self.n_chan)
        self.f_c = f_c
        self.subfractal = subfractal
        self.join_masks = self.f_c.join_masks

    def auto_expand(self):
        ''' Apply a fractal expansion without introducing new weight layers.
        For neuroevolution or inference.'''
        self.intracol_share = False
        self.f_c = self.subfractal(self, self.f_c, n_rec=self.n_recs, n_chan=self.n_chan)
        setattr(self, 'fixed_{}'.format(self.n_recs), None)
        self.f_c.copy_child_weights()
        self.f_c.fixed = copy.deepcopy(self.f_c.fixed)
        self.n_recs += 1
        self.n_cols += 1
        self.f_c.auto_expand()

    def forward(self, x, rnn_hxs=None, masks=None):
        x = self.embed_chan(x)
        depth = pow(2, self.n_recs - 1)
        # (column, join depth)
        if self.rule == 'wide1':
            net_coords = (0, self.n_recs - 1)
        else:
            net_coords = (self.n_recs - 1, depth - 1)
        x = F.relu(self.f_c(x))
        return x

    def clear_join_masks(self):
        ''' Returns a set of join masks that will result in activation flowing
        through the entire fractal network.'''
        if self.rule == 'wide1':
            self.join_masks.fill(1)
            return
        i = 0
        for mask in self.join_masks:
            n_ins = len(mask)
            mask = [1] * n_ins
            self.join_masks[i] = mask
            i += 1

    def set_active_column(self, a):
        ''' Returns a set of join masks that will result in activation flowing
        through a (set of) sequential 'column(s)' of the network.
        - a: an integer, or list of integers, in which case multiple sequential
            columns are activated.'''
        self.global_drop = True
        self.local_drop = False
        if a == -1:
            self.f_c.reset_join_masks(True)
        else:
            self.f_c.reset_join_masks(False)
            self.f_c.set_active_column(a)

    # print('set active col to {}\n{}'.format(a, self.f_c.get_join_masks()))

    def set_local_drop(self):
        self.global_drop = False
        self.active_column = None
        reach = False  # whether or not there is a path thru
        reach = self.f_c.set_local_drop(force=True)
        # print('local_drop\n {}'.format(self.get_join_masks()))
        assert reach

    def set_global_drop(self):
        a = np.random.randint(0, self.n_recs)
        self.set_active_column(a)

    def set_drop_path(self):
        if np.random.randint(0, 2) == 1:
            self.local_drop = self.set_local_drop()
        else:
            self.global_drop = self.set_global_drop()

    def get_join_masks(self):
        return self.f_c.get_join_masks()


class SkipFractal(nn.Module):
    ''' Like fractal net, but where the longer columns compress more,
    and the shallowest column not at all.
    -skip_body - layer or sequence of layers, to be passed through Relu here'''

    def __init__(self, root, f_c, n_rec, skip_body=None):
        '''
        - root: The NN module containing the fractal structure. Has all unique layers.
        - f_c: the previous iteration of this fractal
        - n_rec: the depth of this fractal, 0 when base case
        '''
        super(SkipFractal, self).__init__()
        self.intracol_share = root.intracol_share
        self.n_rec = n_rec
        root.n_recs += 1
        root.n_recs += 1
        root.n_col = 2 * root.n_col
        self.n_chan = 32
        self.f_c = f_c
        self.active_column = root.active_column
        self.join_masks = root.join_masks
        self.global_drop = root.global_drop

        if not self.intracol_share:
            self.fixed = init(nn.Conv2D(self.n_chan, self.n_chan,
                                        3, 1, 1))
            if n_rec > 0:
                self.join = init(nn.Conv2D(self.n_chan * 2, self.n_chan,
                                           3, 1, 1))
                self.up = init(nn.ConvTranspose2D(self.n_chan, self.n_chan,
                                                  2, 2, 0))
                self.dwn = init(nn.ConvTranspose2D(self.n_chan, self.n_chan,
                                                   2, 2, 0))
        else:
            if root.SHARED:
                j = 0  # default index for shared layers
            else:
                j = n_rec  # layer index = recursion index
            if n_rec == 0:
                self.fixed = getattr(root, 'fixed_{}'.format(j))
            if n_rec > 0:
                self.join = getattr(root, 'join_{}'.format(j))
                self.up = getattr(root, 'up_{}'.format(j))
                self.dwn = getattr(root, 'dwn_{}'.format(j))
        if f_c is not None:
            self.skip = f_c.mutate_copy(root)
            self.body = f_c

    def forward(self, x, net_coords=None):
        # print('entering {}'.format(net_coords))
        if x is None:
            return None
        col = net_coords[0]
        depth = net_coords[1]
        x_b, x_a = x, x
        if self.n_rec > 0:
            x_a = self.skip(x_a, (col + 2 ** (self.n_rec - 1), depth - 1))
        else:
            x_a = None
        if self.join_masks[depth][col]:
            # print('including body at: {}'.format(net_coords))
            if self.n_rec > 0:
                x_b = F.relu(self.dwn(x_b))
                x_b = self.body(x_b, (col, depth - 1))
                # print('x_b : \n' + str(x_b))
                if x_b is not None:
                    x_b = F.relu(self.up(x_b))
            else:
                x_b = self.body(x_b)
                return x_b
        else:
            # print('excluding body at: {}'.format(net_coords))
            # print(x_a, x_b)
            x_b = None
        if x_a is None:
            return x_b
        if x_b is None:
            return x_a
        x = F.relu(self.join(th.cat((x_a, x_b), dim=1)))
        # x = x_a + x_b
        return x

    def mutate_copy(self, root):
        ''' In the skip-squeeze fractal, the previous iteration is duplicated and run in parallel.
        The left twin is to be sandwhiched between two new compressing/decompressing layers.
        This function creates the right twin and mutates it, recursively,
        replacing every application of the 'fixed' layer with two in sequence.
        - root: the fractal's owner
        '''
        if self.f_c is not None:
            f_c = self.f_c.mutate_copy(root)
            twin = SkipFractal(root, f_c, self.n_rec)
            return twin
        else:
            twin = SkipFractal(root, None, 0)
            ##win.body = nn.Sequential(twin.body, twin.body)
            return twin


class SubFractal(nn.Module):
    '''
    The recursive part of the network.
    '''

    def __init__(self, root, f_c, n_rec, n_chan):
        super(SubFractal, self).__init__()
        self.n_recs = root.n_recs
        self.n_rec = n_rec
        self.n_chan = n_chan
        self.join_layer = False
        init_ = root.c_init_
        if f_c is not None:
            self.f_c_A = f_c
            if root.intercol_share:
                self.copy_child_weights()
            self.f_c_B = f_c.mutate_copy(root)
            self.join_masks = {'body': True, 'skip': True}
        else:
            self.join_masks = {'body': False, 'skip': True}
        self.active_column = root.active_column
        if (not root.intercol_share) or self.n_rec == 0:
            self.fixed = init_(nn.Conv2d(self.n_chan, self.n_chan, 3, 1, 1))
            if self.join_layer and n_rec > 0:
                self.join = init_(nn.Conv2d(self.n_chan * 2, self.n_chan, 3, 1, 1))

                # if self.join_layer and n_rec > 0:
            #    self.join = getattr(root, 'join_{}'.format(j))

    def auto_expand(self):
        '''just increment n_recs'''
        self.n_recs += 1

    def mutate_copy(self, root):
        ''' Return a copy of myself to be used as my twin.'''
        if self.n_rec > 0:
            f_c = self.f_c_A.mutate_copy(root)
            twin = SubFractal(root, f_c, self.n_rec, n_chan=self.n_chan)
        else:
            twin = SubFractal(root, None, 0, n_chan=self.n_chan)
        if root.intracol_share:
            twin.fixed = self.fixed
        return twin

    def copy_child_weights(self):
        ''' Steal our child's weights to use as our own. Not deep (just refers to existing weights).'''
        if self.n_rec > 0:
            self.fixed = self.f_c_A.fixed
            if self.join_layer:
                self.join = self.f_c_A.join

    def reset_join_masks(self, val=True):
        self.join_masks['skip'] = val
        if self.n_rec > 0:
            self.join_masks['body'] = val
            self.f_c_A.reset_join_masks(val)
            self.f_c_B.reset_join_masks(val)
        else:
            self.join_masks['body'] = False  # not needed

    def set_local_drop(self, force):
        ''' Returns True if path from source to target is yielded to self.join_masks.
                - force: a boolean, whether or not to force one path through.'''
        reach = False
        if self.n_rec == 0:
            self.set_child_drops(False, [0, 1])
            reach = True
        else:
            # try for natural path to target
            prob_body = 1 - (1 / 2) ** self.n_rec
            prob_skip = 1 / 2
            mask = (np.random.random_sample(2) > [prob_body, prob_skip]).astype(int)
            reach = self.set_child_drops(False, mask)
            if not reach and force:  # then force one path down
                mask[1] = np.random.randint(0, 1) <= 1 / (self.n_recs - self.n_rec)
                mask[0] = (mask[1] + 1) % 2
                assert self.set_child_drops(True, mask) == True
                reach = True
        return reach

    def set_child_drops(self, force, mask):
        reach = False
        if force:
            assert 1 in mask
        if mask[1] == 1:
            self.join_masks['skip'] = True
            reach = True
        else:
            self.join_masks['skip'] = False
        self.join_masks['body'] = False
        if mask[0] == 1:
            reach_a = self.f_c_A.set_local_drop(force)
            if reach_a:
                reach_b = self.f_c_B.set_local_drop(force)
                if reach_b:
                    self.join_masks['body'] = True
                    reach = True
            else:
                assert not force
        if force:
            assert reach
        return reach

    def set_active_column(self, col_n):
        if col_n == self.n_rec:
            self.join_masks['skip'] = True
            self.join_masks['body'] = False
        else:
            self.join_masks['skip'] = False
            self.join_masks['body'] = True
            self.f_c_A.set_active_column(col_n)
            self.f_c_B.set_active_column(col_n)

    def get_join_masks(self):
        ''' for printing! '''
        stri = ''
        indent = ''
        for i in range(self.n_recs - self.n_rec):
            indent += '    '
        stri = stri + indent + str(self.join_masks)
        if self.n_rec != 0:
            stri = stri + '\n' + str(self.f_c_A.get_join_masks()) + '\n' + str(self.f_c_B.get_join_masks())
        return stri

    def forward(self, x):
        if x is None: return None
        x_c, x_c1 = x, x

        if self.join_masks['skip']:
            for i in range(1):
                x_c1 = F.relu(
                    # self.dropout_fixed
                    (self.fixed(x_c1)))
        if self.n_rec == 0:
            return x_c1
        if self.join_masks['body']:
            x_c = self.f_c_A(x_c)
            x_c = self.f_c_B(x_c)
        if x_c1 is None:
            return x_c
        if x_c is None:
            return x_c1
        if self.join_layer:
            x = F.relu(
                # self.dropout_join
                (self.join(th.cat((x_c, x_c1), dim=1))))
        else:
            x = (x_c1 + x_c * (self.n_rec)) / (self.n_rec + 1)
        return x


class SubFractal_squeeze(nn.Module):
    def __init__(self, root, f_c, n_rec, net_coords=None):
        super(SubFractal_squeeze, self).__init__()
        self.map_width = root.map_width
        self.n_rec = n_rec
        root.n_recs += 1
        self.n_chan = root.n_chan
        self.join_masks = root.join_masks
        self.active_column = root.active_column
        self.num_down = min(int(math.log(self.map_width, 2)) - 1, n_rec)
        self.dense_nug = (self.num_down > 1)
        self.join_layer = False
        self.intracol_share = root.intracol_share
        self.init_ = init_ = lambda m: init(m,
                                            nn.init.dirac_,
                                            lambda x: nn.init.constant_(x, 0.1),
                                            nn.init.calculate_gain('relu'))

        if root.intercol_share:
            j = 0
        else:
            j = n_rec
        if self.intracol_share:
            # for i in range(self.num_down):
            for i in range(1):
                setattr(self, 'dwn_{}'.format(i),
                        init_(nn.Conv2d(self.n_chan, self.n_chan,
                                        2, 2, 0)))
                setattr(self, 'up_{}'.format(i),
                        init_(nn.ConvTranspose2d(self.n_chan,
                                                 self.n_chan, 2, 2, 0)))
            self.fixed = init_(nn.Conv2d(self.n_chan,
                                         self.n_chan, 3, 1, 1))
        elif not self.intracol_share:
            if n_rec > 0:
                self.up = getattr(root, 'up_{}'.format(j))
                self.dwn = getattr(root, 'dwn_{}'.format(j))
                if self.join_layer:
                    self.join = getattr(root, 'join_{}'.format(j))
            # if self.dense_nug:
            #     squish_width = self.map_width / (2 ** self.num_down)
            #     hidden_size = int(squish_width * squish_width * self.n_chan)
            #     linit_ = lambda m: init(m,
            #         nn.init.orthogonal_,
            #         lambda x: nn.init.constant_(x, 0))
            #     self.dense = linit_(nn.Linear(hidden_size, hidden_size))
            self.fixed = getattr(root, 'fixed_{}'.format(j))
            if root.batch_norm:
                self.bn_join = nn.BatchNorm2d(self.n_chan)
                if self.num_down == 0:
                    setattr(self, 'bn_fixed_{}'.format(0), nn.BatchNorm2d(self.n_chan))
                for i in range(self.num_down):
                    setattr(self, 'bn_dwn_{}'.format(i), nn.BatchNorm2d(self.n_chan))
                    setattr(self, 'bn_fixed_{}'.format(i), nn.BatchNorm2d(self.n_chan))
                    setattr(self, 'bn_up_{}'.format(i), nn.BatchNorm2d(self.n_chan))
        self.f_c_A = f_c
        if f_c is not None:
            self.f_c_B = f_c.mutate_copy(root)
        else:
            self.f_c_B = f_c

    def mutate_copy(self, root):
        ''' '''
        if self.f_c_A is not None:
            f_c = self.f_c_A.mutate_copy(root)
            twin = SubFractal_squeeze(root, f_c, self.n_layer)
            return twin
        else:
            twin = SubFractal_squeeze(root, None, 0)
            ##win.body = nn.Sequential(twin.body, twin.body)
            return twin

    def forward(self, x, net_coords):
        if x is None:
            return x
        x_c, x_c1 = x, x
        col = net_coords[0]
        depth = net_coords[1]
        if self.n_rec > 0:
            x_c = self.f_c_A(x_c, (col - 1, depth - 2 ** (col - 1)))
            x_c = self.f_c_B(x_c, (col - 1, depth))
            if self.join_masks[depth][col]:
                for d in range(self.num_down):
                    # bn_dwn = getattr(self, 'bn_dwn_{}'.format(d))
                    dwn = getattr(self, 'dwn_{}'.format(0))
                    x_c1 = F.relu(  # bn_dwn
                        (dwn(x_c1)))
                    # if self.dense_nug:
                    #     x_c1_shape = x_c1.shape
                    #     x_c1 = x_c1.view(x_c1.size(0), -1)
                    #     x_c1 = F.tanh(self.dense(x_c1))
                    #     x_c1 = x_c1.view(x_c1_shape)
                for f in range(1):
                    # bn_fixed= getattr(self, 'bn_fixed_{}'.format(f))
                    x_c1 = F.relu(  # bn_fixed
                        (self.fixed(x_c1)))
                for u in range(self.num_down):
                    # bn_up = getattr(self, 'bn_up_{}'.format(u))
                    up = getattr(self, 'up_{}'.format(0))
                    x_c1 = F.relu(  # bn_up
                        up(x_c1, output_size=(x_c1.size(0), x_c1.size(1),
                                              x_c1.size(2) * 2, x_c1.size(3) * 2)))
        if x_c is None or col == 0:
            return x_c1
        if x_c1 is None:
            return x_c
        if self.join_layer:
            x = F.relu(  # self.bn_join
                (self.join(th.cat((x_c, x_c1), dim=1))))
        else:
            x = (x_c1 + x_c * self.n_rec) / (self.n_rec + 1)
        return x


class CustomNetwork(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Shared network (empty in this case)
        self.shared_net = nn.Sequential(nn.Conv2d(feature_dim, 32, kernel_size=5),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, kernel_size=3),
                                        nn.ReLU())

        # Policy/Action network
        self.policy_net = nn.Sequential(
            nn.Conv2d(64, last_layer_dim_pi, kernel_size=1),
            nn.ReLU()
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.Tanh(),
            nn.Linear(256, last_layer_dim_vf),
            nn.ReLU()
        )

        self.features_extractor = CombinedExtractor

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        flattened_features = self.features_extractor(features)
        policy_features = self.shared_net(flattened_features)
        return self.policy_net(policy_features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        flattened_features = self.features_extractor(features)
        value_features = self.shared_net(flattened_features)
        return self.value_net(value_features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        f_d = self.features_dim
        self.mlp_extractor = CustomNetwork(f_d)
