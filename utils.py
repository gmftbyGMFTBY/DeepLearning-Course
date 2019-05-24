#!/usr/bin/python

import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output


def show_state(env, step=0, info=""):
    # info may contains the reward
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title(f"{env.spec.id} | Step: {step} | Last Reward: {info}")
    plt.axis('off')

    clear_output(wait=True)
    display.display(plt.gcf())


if __name__ == "__main__":
    pass