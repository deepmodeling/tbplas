#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


a = 0.142
t = 3.0


def plot_dp():
    prefix = ["lind", "tbpm"]
    colors = ["r", "b"]
    for i, pref in enumerate(prefix):
        omegas = np.load(f"{pref}/omegas_dp.npy")
        data = np.load(f"{pref}/dp.npy")
        plt.plot(omegas/t, -data.imag[0]*t*a**2, color=colors[i])
    plt.grid()
    plt.show()
    plt.close()


def plot_eps():
    prefix = ["lind", "tbpm"]
    colors = ["r", "b"]
    for i, pref in enumerate(prefix):
        omegas = np.load(f"{pref}/omegas_eps.npy")
        data = np.load(f"{pref}/eps.npy")
        plt.plot(omegas, data[0].real, color=colors[i])
    plt.minorticks_on()
    plt.grid()
    plt.show()
    plt.close()


def plot_ac():
    prefix = ["lind", "tbpm"]
    colors = ["r", "b"]
    for i, pref in enumerate(prefix):
        omegas = np.load(f"{pref}/omegas_ac.npy")
        data = np.load(f"{pref}/ac.npy")
        if pref == "tbpm":
            data = data[0]
        plt.plot(omegas/t, data.real*4, color=colors[i])
    plt.minorticks_on()
    plt.grid()
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_dp()
    plot_eps()
    plot_ac()
