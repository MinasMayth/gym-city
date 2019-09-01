# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})
import numpy as np

from imutils import paths

def create_gif(inputPath, outputPath, delay, finalDelay, loop):
	# grab all image paths in the input directory
	imagePaths = sorted(list(paths.list_images(inputPath)))
	# remove the last image path in the list
	lastPath = imagePaths[-1]
	imagePaths = imagePaths[:-1]
	# construct the image magick 'convert' command that will be used
	# generate our output GIF, giving a larger delay to the final
	# frame (if so desired)
	cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(
		delay, " ".join(imagePaths), finalDelay, lastPath, loop,
		outputPath)
	os.system(cmd)

def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size, col=None):
    datas = []
    if col is not None:
        infiles = glob.glob(os.path.join(indir, 'col_{}_eval.csv'.format(col)))
    else:
        infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))
    if len(infiles) == 0:
        print('no files found at {}'.format(indir))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        if len(result) > 2:
            bin_size = len(result) # hack, so we see graphs asap
        else:
            return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

class Plotter(object):
    def __init__(self, n_cols, indir, n_proc):
        self.n_cols = n_cols + 1
        self.avgs = np.zeros((n_cols + 1))
        self.n_samples = np.zeros((n_cols + 1)) # how many episodes per process, 
        # this may be different for each column due to interrupted evaluation
        self.n_proc = n_proc # how many processes
        self.indir = indir


    def get_col_avg(self, col=None):
        if col is not None:
            infiles = glob.glob(os.path.join(self.indir, 'col_{}_eval.csv'.format(col)))
        else:
            infiles = glob.glob(os.path.join(self.indir, '*.monitor.csv'))
        if len(infiles) == 0:
            print('no files found at {}'.format(self.indir))

        i = 0
        net_reward = 0
        for inf in infiles:
            with open(inf, 'r') as f:
                f.readline()
                f.readline()
                for line in f:
                    tmp = line.split(',')
                    r = float(tmp[0])
                    net_reward += r
                    i += 1
        if i != 0:
            avg_reward = net_reward / i
        else:
            avg_reward = 0
        self.avgs[col] = avg_reward
        return avg_reward

    def get_col_std(self, col=None):
        if col is not None:
            infiles = glob.glob(os.path.join(self.indir, 'col_{}_eval.csv'.format(col)))
        else:
            infiles = glob.glob(os.path.join(self.indir, '*.monitor.csv'))
        if len(infiles) == 0:
            print('no files found at {}'.format(self.indir))

        mean = self.avgs[col]
        i = 0
        net_deviation = 0
        for inf in infiles:
            with open(inf, 'r') as f:
                f.readline()
                f.readline()
                for line in f:
                    tmp = line.split(',')
                    r = float(tmp[0])
                    net_deviation += np.abs(mean - r)
                    i += 1
        if i != 0:
            avg_deviation = net_deviation / i
        else:
            avg_deviation = 0
        self.avgs[col] = avg_deviation
        return avg_deviation



    def bar_plot(self, viz, win, folder, game, name, num_steps, n_cols=None):
        fig = plt.figure()
        x = [i for i in range(-1, n_cols)]

        h = [self.get_col_avg(col = i) for i in range(-1, n_cols)]
        e = [self.get_col_std(col = i) for i in range(-1, n_cols)]
        plt.bar(x, h, yerr=e, color=color_defaults[:n_cols + 1])

        plt.xlabel('Columns')
        plt.ylabel('Rewards')

        plt.title(game)
        plt.legend(loc=4)
        figfolder = folder.replace('/logs_eval_', '/eval_')
       #figfolder = folder.replace('/logs', '/train_')
        print('should be saving graph now as {}'.format(figfolder))
        plt.savefig('{}/bar_fig.png'.format(figfolder), format='png')
        plt.show()
        plt.draw()

        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        plt.close(fig)

        # Show it in visdom
        image = np.transpose(image, (2, 0, 1))
        return viz.image(image, win=win)

    def visdom_plot(self, viz, win, folder, game, name, num_steps, bin_size=100, smooth=5, n_graphs=None,
            x_lim=None, y_lim=None):
        if folder.endswith('logs'):
            evl = False
        elif folder.endswith('logs_eval'):
            evl = True
        tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
        ticks = tick_fractions * num_steps
        tick_names = ["{:.0e}".format(tick) for tick in ticks]
        fig = plt.figure()
        if n_graphs is not None:
            #print('indaplotter')
            color = 0
            for i in n_graphs:
                tx, ty = load_data(folder, smooth, bin_size, col=i)
                if tx is None or ty is None:
                    #print('could not find x y data columns in csv')
                    pass
                   #return win

                else:
                    plt.plot(tx, ty, label="col_{}".format(i), color=color_defaults[color])
                    color += 1
        else:
            tx, ty = load_data(folder, smooth, bin_size)
            if tx is None or ty is None:
                return win
            if evl:
                color = 3
                plt.plot(tx, ty, label='det-eval', color=color_defaults[color])
            else:
                plt.plot(tx, ty, label="non-det")


        plt.xticks(ticks, tick_names)
        if x_lim:
            plt.xlim(*x_lim)
        else:
            plt.xlim(0, num_steps * 1.01)
        if y_lim:
            plt.ylim(*y_lim)

        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')

        plt.title(game)
        plt.legend(loc=4)
        if evl:
            figfolder = folder.replace('/logs_eval', '/eval_')
        else:
            figfolder = folder.replace('/logs', '/train_')
        print('should be saving graph now as {}'.format(figfolder))
        plt.savefig('./{}fig.png'.format(figfolder), format='png')
        plt.show()
        plt.draw()

        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        plt.close(fig)

        # Show it in visdom
        image = np.transpose(image, (2, 0, 1))
        return viz.image(image, win=win)

def man_eval_plot(indir, n_cols=5, num_steps=200000000, n_proc=96, x_lim=None, y_lim=None):
    plotter = Plotter(n_cols=n_cols, indir=indir, n_proc=n_proc)
    from visdom import Visdom
    viz = Visdom()
    win = None
    win = plotter.visdom_plot(viz, win, "{}/logs_eval".format(indir), "",  "Fractal Net", num_steps=num_steps, 
        n_graphs=range(-1,n_cols), x_lim=x_lim, y_lim=y_lim)
    return win

if __name__ == "__main__":
    from visdom import Visdom
    import argparse
    viz = Visdom()
    win = None
    parser = argparse.ArgumentParser(description='viz')
    parser.add_argument('--load-dir', default=None,
            help='directory from which to load agent logs (default: ./trained_models/)')
    visdom_plot(viz, None, '/tmp/gym/', 'BreakOut', 'a2c', bin_size=100, smooth=1)
