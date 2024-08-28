import math
import time
import os
os.environ['HOTSPICE_USE_GPU'] = 'False'

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation, cm, colormaps, colors

from examples import examplefunctions as ef
try: from context import hotspice
except ModuleNotFoundError: import hotspice

# Set ASI parameters
a = 280e-9 # [m]
θ = 70 # [deg]
b = a * np.tan(θ*np.pi/180) # [m]

T = 2250 # [K]
E_B = 5e-22 # [J]
n = 6 *(4+1) # Multiple of 4 + 1
moment = None # Default
angle = np.pi/2 # [rad]

pattern = "uniform"
energies = [hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()]

# Set experimental method
N = 10e3 # Number of switches
repeat = 1


def animate_relaxation(mm: hotspice.Magnets, animate=1, speed=2, N=N, save=False):
    mm.history.clear()
    mm.t=0

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(6, 4.8))
    ax1 = fig.add_subplot(111)
    cmap = colormaps['hsv']

    # Assuming ASI in plane
    extent = np.array([mm.x_min - mm.dx / 2, mm.x_max + mm.dx / 2, mm.y_min - mm.dy / 2, mm.y_max + mm.dy / 2])
    nonzero = mm.nonzero
    mx, my = (np.multiply(mm.m, mm.orientation[:, :, 0])[nonzero]), (
    np.multiply(mm.m, mm.orientation[:, :, 1])[nonzero])
    ax1.quiver((mm.xx[nonzero]), (mm.yy[nonzero]), mx, my, color=cmap((np.arctan2(my, mx) / 2 / np.pi) % 1),
               pivot='mid', scale=1.1 / mm._get_closest_dist(), headlength=17, headaxislength=17, headwidth=7,
               units='xy')
    ax1.set_aspect('equal')
    ax1.set_xlim(extent[:2])
    ax1.set_ylim(extent[2:])

    # This is the function that gets called each frame
    def animate_relaxation_update(i):
        currStep = i * speed
        for j in range(currStep, currStep + speed):
            mm.update()
            mm.history_save()
            fig.suptitle(f"Time {mm.t}\u2009s")
            mx, my = (np.multiply(mm.m, mm.orientation[:, :, 0])[nonzero]), (np.multiply(mm.m, mm.orientation[:, :, 1])[nonzero])
            ax1.quiver((mm.xx[nonzero]), (mm.yy[nonzero]), mx, my, color=cmap((np.arctan2(my, mx) / 2 / np.pi) % 1),
                           pivot='mid', scale=1.1 / mm._get_closest_dist(), headlength=17, headaxislength=17, headwidth=7, units='xy')

    # Assign the animation to a variable, to prevent it from getting garbage-collected
    t = time.perf_counter()
    anim = animation.FuncAnimation(fig, animate_relaxation_update,
                                    frames=int(N//speed)*max(1, save), interval=speed/animate,
                                    blit=False, repeat=True)

    if save: # TODO: cannot save because file doesn't exist??
        mywriter = animation.FFMpegWriter(fps=30)
        if not os.path.exists("videos"): os.makedirs("videos")
        anim.save(f"videos/{type(mm).__name__}_{mm.nx}x{mm.ny}_t{0}-{mm.t}_N{N}x{save}.mp4", writer=mywriter, dpi=300)
        print(f"Performed {mm.switches} switches in {time.perf_counter() - t:.3f} seconds.")

    plt.show()
    print(f"Performed {mm.switches} switches in {time.perf_counter() - t:.3f} seconds.")


def relax(mm: hotspice.Magnets, N=None, data=None, turn=None, **kwargs):
    """
        Relaxes ASI from an initial configuration by calling mm.update() and returns data.
    """
    # Track program runtime
    t = time.perf_counter()
    n_start = mm.switches

    for i in range(int(N)):
        # Update one Néel step
        mm.update()
        mm.history_save()

        # Save mm.t, AFMness, E_tot, m_avg to results
        data[i][0][turn] = mm.t
        data[i][1][turn] = hotspice.plottools.get_AFMness(mm)
        data[i][2][turn] = mm.E_tot
        data[i][3][turn] = mm.m_avg

    dt = time.perf_counter() - t
    print(f"Simulated {mm.switches - n_start:.0f} switches ({N:.0f} steps on {mm.m.shape[0]:.0f}x{mm.m.shape[1]:.0f} grid) in {dt:.3f} seconds.")
    #hotspice.plottools.show_m(mm, **kwargs) # Show relaxed state of ASI

    return data


def plot_relaxation(data=None, turn=None):
    fig,ax = plt.subplots(3, 1)
    for i in range(3):
        ax[i].plot(data[:,0,j], data[:,i+1,j])
        ax[i].set_xlabel("Time")
    ax[0].set_ylabel("AFM_ness")
    ax[1].set_ylabel("Total energy [J]")
    ax[2].set_ylabel("Average magnetization")
    fig.suptitle(f"Relaxation of ASI on run {turn+1}.")
    plt.show()


def analyze_data(data=None):
    """
    Takes data from ASI relaxation and returns:
        results = [mean(final m_avg), var(final m_avg), mean(relaxation time), var(relaxation time)]
    """
    results[0] = np.mean(data[int(N)-1,3,:])
    results[1] = np.var(data[int(N)-1,3,:])

    rt = np.zeros(repeat)
    for j in range(repeat):
        for t in range(int(N)):
            if math.isclose(data[t][3][j], data[0,3,j]/np.e, rel_tol=1e-2):
                rt[j] = data[t][0][j]
                break
    results[2] = np.mean(rt)
    results[3] = np.var(rt)
    print(results)

    return results


if __name__ == "__main__":
    # Initialise empty outputs
    data = (np.zeros((int(N),4,repeat)))
    results = np.zeros(4)
    ground_state = None

    for j in range(repeat):

        mm = hotspice.ASI.IP_Ising_Offset(a=a, b=b, n=n, T=T, E_B=E_B, angle=angle,
                                          pattern=pattern, energies=energies, PBC=True)  # Create ASI input

        hotspice.plottools.show_lattice(mm)
        data = relax(mm, N=N, data=data, turn=j)
        plot_relaxation(data, turn=j)

    results = analyze_data(data)