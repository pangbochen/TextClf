# -*- coding: UTF-8 -*-

import visdom
import time
import numpy as np

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        :param d: dict(name, value) ('loss', 0.11)
        :return:

        '''
        for k, v in d.items():
            self.plot(k, v)

    def plot(self, name, y, **kwargs):
        # self.plot('loss', 1.00)
        x = self.index.get(name, 0)
        self.vis.line(
            Y=np.array([y]), X=np.array([x]),
            win=name,
            opts=dict(title=name),
            update=None if x == 0 else 'append',
            **kwargs
        )
        self.index[name] = x + 1

    def log(self, info, win='log_text'):
        '''self.log({'loss':1, 'lr':0.0001})'''
        self.log_text += ('[{time}]{info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info
        ))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name=name)
