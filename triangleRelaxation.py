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



## Parameters
T = 300 # [K]
E_B = 5e-22 # [J]
nx = 5 *4+1 # Multiple of 4 + 1



def triangle_test():
    mm = hotspice.ASI.IP_Ising(4e-6, nx, T=T, E_B=E_B, pattern='AFM',
                                  energies=[hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()], PBC=False)
    # hotspice.gui.show(mm)

    hotspice.plottools.show_m(mm)
    datastream = hotspice.io.RandomBinaryDatastream()
    fi = hotspice.io.FieldInputter(datastream=datastream, angle=theta)

    fi.input_single(mm, 1)
    hotspice.plottools.show_m(mm)

    fi.input_single(mm=mm, value=0)
    hotspice.plottools.show_m(mm)

    ef.run_a_bit(mm, N=50e3)



def animate_relaxation(mm: hotspice.Magnets, quiver=True, animate=1, speed=2, N=200000, t_s=0, t_f=250, save=False, fill=False, avg=True, pattern=None):
    if pattern:
        mm.initialize_m(pattern)

    mm.history.clear()
    mm.t = 0
    #AFM_ness = []
    #hotspice.core.SimParams.UPDATE_SCHEME = "Néel"

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(6, 4.8))
    ax1 = fig.add_subplot(111)
    cmap = colormaps['hsv']
    if mm.in_plane:
        if quiver:
            extent = np.array([mm.x_min - mm.dx / 2, mm.x_max + mm.dx / 2, mm.y_min - mm.dy / 2, mm.y_max + mm.dy / 2])
            nonzero = mm.nonzero
            mx, my = (np.multiply(mm.m, mm.orientation[:, :, 0])[nonzero]), (np.multiply(mm.m, mm.orientation[:, :, 1])[nonzero])
            ax1.quiver((mm.xx[nonzero]), (mm.yy[nonzero]), mx, my, color=cmap((np.arctan2(my, mx) / 2 / np.pi) % 1),
                       pivot='mid', scale=1.1 / mm._get_closest_dist(), headlength=17, headaxislength=17, headwidth=7, units='xy')
            ax1.set_aspect('equal')
            ax1.set_xlim(extent[:2])
            ax1.set_ylim(extent[2:])
        if quiver==False:
            avg = hotspice.plottools.Average.resolve(avg, mm)
            h = ax1.imshow(hotspice.plottools.get_rgb(mm, fill=fill, avg=avg),
                           cmap=cmap, origin='lower', vmin=0, vmax=math.tau,
                           extent=hotspice.plottools._get_averaged_extent(mm, avg))
            c1 = plt.colorbar(h)
            c1.ax.get_yaxis().labelpad = 30
            c1.ax.set_ylabel(
                f"Averaged magnetization angle [rad]\n('{avg.name.lower()}' average{', PBC' if mm.PBC else ''})",
                rotation=270, fontsize=12)
    else:
        avg = hotspice.plottools.Average.resolve(avg, mm)
        r0, g0, b0, _ = cmap(.5) # Value at angle 'pi' (-1)
        r1, g1, b1, _ = cmap(0) # Value at angle '0' (1)
        cdict = {'red':   [[0.0,  r0, r0], # x, value_left, value_right
                   [0.5,  0.0, 0.0],
                   [1.0,  r1, r1]],
         'green': [[0.0,  g0, g0],
                   [0.5, 0.0, 0.0],
                   [1.0,  g1, g1]],
         'blue':  [[0.0,  b0, b0],
                   [0.5,  0.0, 0.0],
                   [1.0,  b1, b1]]}
        newcmap = colors.LinearSegmentedColormap('OOP_cmap', segmentdata=cdict, N=256)
        h = ax1.imshow(hotspice.plottools.get_rgb(mm, fill=fill, avg=avg),
                       cmap=newcmap, origin='lower', vmin=-1, vmax=1, extent=hotspice.plottools._get_averaged_extent(mm, avg))
        c1 = plt.colorbar(h)
        c1.ax.get_yaxis().labelpad = 30
        c1.ax.set_ylabel(f"Averaged magnetization\n('{avg.name.lower()}' average{', PBC' if mm.PBC else ''})", rotation=270, fontsize=12)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    fig.suptitle(f"Time {mm.t:.3f}\u2009s") # \u2009 is a small space similar to \, in LaTeX

    # This is the function that gets called each frame
    def animate_relaxation_update(i):
        currStep = i*speed
        for j in range(currStep, currStep + speed):
            if mm.t <= t_f:
                #mm._minimize_all()
                mm.update()
                mm.t = mm.t + 1
                #AFM_ness.append(hotspice.plottools.get_AFMness(mm))
                mm.history_save()
            else:
                return
            fig.suptitle(f"Time {mm.t:.3f}\u2009s")
        if quiver:
            mx, my = (np.multiply(mm.m, mm.orientation[:, :, 0])[nonzero]), (np.multiply(mm.m, mm.orientation[:, :, 1])[nonzero])
            ax1.quiver((mm.xx[nonzero]), (mm.yy[nonzero]), mx, my, color=cmap((np.arctan2(my, mx) / 2 / np.pi) % 1),
                           pivot='mid', scale=1.1 / mm._get_closest_dist(), headlength=17, headaxislength=17, headwidth=7, units='xy')
        if quiver==False: # Animate averaged colours
            h.set_array(hotspice.plottools.get_rgb(mm, fill=fill, avg=avg))
            return h, # This has to be an iterable!

    # This function clears previous frames
    # TODO: make it work!!!!
    def init():
        ax1.clear()
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        fig.suptitle(f"Time {mm.t:.3f}\u2009s")  # \u2009 is a small space similar to \, in LaTeX

    # Assign the animation to a variable, to prevent it from getting garbage-collected
    t = time.perf_counter()
    anim = animation.FuncAnimation(fig, animate_relaxation_update,
                                    frames=int(N//speed)*max(1, save), init_func=init, interval=speed/animate,
                                    blit=False, repeat=True)
    if save:
        mywriter = animation.FFMpegWriter(fps=30)
        if not os.path.exists("videos"): os.makedirs("videos")
        anim.save(f"videos/{type(mm).__name__}_{mm.nx}x{mm.ny}_t{t_s}-{t_f}_N{N}x{save}.mp4", writer=mywriter, dpi=300)
        print(f"Performed {mm.switches} switches in {time.perf_counter() - t:.3f} seconds.")

    plt.show()
    #hotspice.plottools.show_history(mm, y_quantity=AFM_ness, y_label="AFM-ness")
    print(f"Performed {mm.switches} switches in {time.perf_counter() - t:.3f} seconds.")



def AFM_pulse(mm: hotspice.Magnets, turn=None, data=None, t_f=None):
    """Called by experiment_repeater."""
    mm.history.clear()
    mm.initialize_m('AFM')
    #AFM_ness = []

    #hotspice.plottools.show_m(mm)
    #print("original AFMness is", hotspice.plottools.get_AFMness(mm)) # Checkpoint

    # Apply (uniform) field
    datastream = hotspice.io.RandomBinaryDatastream(p0=0)
    fi = hotspice.io.FieldInputter(datastream=datastream, angle=45)
    #fi.input_single(mm, 1)
    #hotspice.plottools.show_m(mm)
    #print("AFMness after 1 field pulse is", hotspice.plottools.get_AFMness(mm)) # Checkpoint

    # Remove field
    fi.input_zero(mm, 0)
    mm.update()
    #hotspice.plottools.show_m(mm)
    #print("AFMness after 0 field pulse is", hotspice.plottools.get_AFMness(mm)) # Checkpoint

    mm.t = 0  # Start time after field pulse

    # Relax ASI over time
    while mm.t < t_f:
        #mm._minimize_all()
        mm.update() # Random thermal fluctuations
        #AFM_ness.append(hotspice.plottools.get_AFMness(mm))
        mm.history_save()

        data[int(mm.t)][0][turn]=hotspice.plottools.get_AFMness(mm)
        data[int(mm.t)][1][turn]=mm.E_tot
        data[int(mm.t)][2][turn]=mm.m_avg

        mm.t = mm.t + 1
        #print(hotspice.plottools.get_AFMness(mm))

    #print("AFMness after relaxation is", hotspice.plottools.get_AFMness(mm)) # Checkpoint
    #hotspice.plottools.show_m(mm)
    #plot_new(data)
    #hotspice.plottools.show_history(mm, y_quantity=AFM_ness, y_label="AFM-ness") # Plot relaxation graphs


# Old code from test_thetaRelaxation - timekeeping is faulty
def relax(mm: hotspice.Magnets, t_f=None, turn=None):
    """
    Relaxes ASI from an initial configuration by calling mm.update() and returns data.
    """
    while mm.t < t_f:
        # Update one Néel step
        mm._minimize_all()
        mm.update()
        mm.history_save()

        # Save mm.t, AFMness, E_tot, m_avg to results
        data[int(mm.t)][0][turn] = mm.t
        data[int(mm.t)][1][turn] = hotspice.plottools.get_AFMness(mm)
        data[int(mm.t)][2][turn] = mm.E_tot
        data[int(mm.t)][3][turn] = mm.m_avg

        mm.t = mm.t + 1

    return data


def experiment_repeater(mm, t_f=200, repeat=1):
    """
    Called by main for a single ASI, or half_life_sweeper for multiple ASIs.
    Repeats an experiment e.g. AFM_pulse, then plots & analyses the resulting dataset.
    """
    data_avg = np.zeros((t_f,3))
    data = np.zeros((t_f,3,repeat))
    for j in range(repeat):
        AFM_pulse(mm, turn=j, data=data, t_f=t_f)
    data_avg = np.mean(data, axis=2)
    hl = half_life(data_avg=data_avg, t_f=t_f)
    plot_new(data_avg, repeat=repeat)
    return hl



def half_life(data_avg=None, t_f=None):
    """
    Called by experiment_repeater.
    Returns array hl where:
        columns are [0]:AFM_ness, [1]:Energy [j], [2]:Average magnetization.
        rows are [0]:'Half-life'-like time variable, [1]:Average of max & min value, [2]:Tolerance of closeness.
    """
    hl = np.zeros((3,3))
    for i in range(np.size(hl,1)):
        hl[1,i] = np.mean([np.max(data_avg[:,i]), np.min(data_avg[:,i])])
        if i == 0 or i == 1: tol = 5
        if i == 2: tol = 9
        while hl[0,i] == 0:
            for t in range(t_f):
                #print(t, data_avg[t,i], hl[1,i], 10**(-1*tol))
                if math.isclose(data_avg[t,i], hl[1,i], rel_tol=10**(-1*tol)):
                    if t==0: hl[0,i]=0.00001
                    else: hl[0,i] = t
                    hl[2,i] = 10**(-1*tol)
                    break
            tol = tol-1
    print(hl)
    return hl


def plot_new(data, repeat=None):
    """Called by experiment_repeater."""
    fig,ax = plt.subplots(len(data[0]),1)
    for i in range(len(data[0])):
        ax[i].plot(data[:,i])
        ax[i].set_xlabel("Time")
    ax[0].set_ylabel("AFM_ness")
    ax[1].set_ylabel("Total energy [J]")
    ax[2].set_ylabel("Average magnetization")
    fig.suptitle(f"ASI properties after uniform field pulse, averaged over {repeat} repeats.")
    plt.show()


def half_life_sweeper(a_range=None, θ_range=None, b_range=None, n_range=None, T_range=None, E_B_range=None, angle_range=None):
    """
        Plots the 'half-life'-like time variable against some parameter of the ASI geometry.
        Six subplots: Each of [0]:AFM_ness, [1]:Energy [j], [2]:Average magnetization has two plots (half-life & mean).
        Parameters include: a, b, n, T, E_B, angle. # TODO: test more complex geometries using occupation in ASI/core
        A parameter is varying by supplying argument x_range = [x_min, x_max, x_steps]
        # TODO: currently only b_range can be varied easily
    """
    # Set values of fixed parameters & create array for varying parameter
    if a_range==None: a = 4e-6
    if b_range==None: b = a * np.sqrt(3)
    else: b = np.linspace(b_range[0], b_range[1], b_range[2])
    if n_range==None: n = 5 *4+1
    if T_range==None: T = 300 # [K]
    if E_B_range==None: E_B = 5e-22 # [J]
    if angle_range==None: angle=np.pi/2 # [rad]

    big_data = np.zeros((b_range[2], 2, 3))
    # Fill big_data with half-lives as a function of the varying parameter
    for i in range(b_range[2]):
        mm = hotspice.ASI.IP_Ising_Offset(a=a, b=b[i], n=n, T=T, E_B=E_B, angle=angle,
                                      pattern='uniform', energies=[hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()], PBC=False)
        hl = experiment_repeater(mm)
        # Takes top two rows from half-life array
        for j in range(3):
            big_data[i,0,j]=hl[0,j] # 'Half-life'-like time variable
            big_data[i,1,j]=hl[1,j] # Average of first & final value

    #print(big_data)

    # Plot final graphs
    fig,ax = plt.subplots(3,2)
    for i in range(3):
        ax[i][0].plot(θ, big_data[:,0,i])
        ax[i][1].plot(θ, big_data[:,1,i])
        ax[i][0].set_xlabel("b")
        ax[i][1].set_xlabel("b")

    ax[0][0].set_ylabel("AFM_ness: half-life [s]")
    ax[0][1].set_ylabel("AFM_ness: mean")
    ax[1][0].set_ylabel("Total energy: half-life [s]")
    ax[1][1].set_ylabel("Total energy: mean [J]")
    ax[2][0].set_ylabel("Average magnetization: half-life [s]")
    ax[2][1].set_ylabel("Average magnetization: mean")
    plt.tight_layout() # Tidy up axes so not overlapping
    plt.show()



if __name__ == "__main__":
    mm = hotspice.ASI.IP_Ising_Offset(a=4e-6, n=nx, T=T, E_B=E_B, pattern='random',
                                  energies=[hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()], PBC=False, angle=np.pi/2)

    #AFM_pulse(mm)
    #hotspice.plottools.show_lattice(mm)
    #hotspice.plottools.show_m(mm)
    animate_relaxation(mm, quiver=True, pattern="uniform", t_f=500)
    #experiment_repeater(mm, t_f=200, repeat=1)
    #half_life_sweeper(b_range=[1e-6, 10e-6, 200])
    # TODO: write program which tests combinations of varying parameters & plots 3D parameter spaces?