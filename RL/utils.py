#!/usr/bin/python

import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output
import pandas
import numpy as np

n_bins = 8
n_bins_angle = 10
cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]


def show_state(env, step=0, info=""):
    # info may contains the reward
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title(f"{env.spec.id} | Step: {step} | Info: {info}")
    plt.axis('off')

    clear_output(wait=True)
    display.display(plt.gcf())
    
def build_state(features):
    # for basic reinforcement learning
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    # for basic reinforcement learning
    return np.digitize(x=[value], bins=bins)[0]

def discretize(state):
    cart_position, pole_angle, cart_velocity, angle_rate_of_change = state
    state = build_state([to_bin(cart_position, cart_position_bins),
                         to_bin(pole_angle, pole_angle_bins),
                         to_bin(cart_velocity, cart_velocity_bins),
                         to_bin(angle_rate_of_change, angle_rate_bins)])
    return state


if __name__ == "__main__":
    pass