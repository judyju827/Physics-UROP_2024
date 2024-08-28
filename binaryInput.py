import os
os.environ['HOTSPICE_USE_GPU'] = 'False'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error

try: from context import hotspice
except ModuleNotFoundError: import hotspice


# Set ASI parameters
a = 280e-9  # [m]
T = 300  # [K]
E_B = 5e-22  # [J]
n = 4 *(4+1)  # Multiple of 4 + 1
Msat = 800e3  # [A/m] Permalloy
V = 4.8e-23  # [m^3]
angle = np.pi/2  # [rad]

pattern = "random"  # Initial config
energies = [hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()]


# Ising_Offset ASI geometry
θ = 70  # [deg]
b = a * np.tan(θ * np.pi / 180)  # [m]


# Set input
binary = False  # Use binary or sinusoidal field
input = [1,1,0,0,1,1,1,1,0,0,0,0,1,1,0,0,1,0,1,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,1,
         0,0,0,0,1,1,0,1,1,1,0,0,0,1,0,0,1,0,1,0,1,1,0,1,1,1,0,0,1,0,0,1,0,0,0,1,1,
         1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,1,0,1,
         1,1,1,1,0,0,0,1,0,1,1,1,1,0,0,1,0,1,0,1,1,0,1,1,0,1,0,1,0,0,0,0,1,1,1,1,0,
         0,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,0,1,0,1,1,0,0,0,0,0,1,0,1,0,1,1,1,1,1,1,0,
         1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,
         0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,
         1,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,1,0,0,0,0,1,1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,
         1,0,0,1,0,1,1,1,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,0,
         0,1,1,0,0,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,0,0,0,1,1,0,1,1,1,0,1,0,1,1,
         0,0,0,0,0,1,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,
         0,1,1,0,1,0,0,0,1,1,1,1,0,1,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,1,1,
         1,0,0,1,1,0,1,1,0,1,1,1,0,1,0,1,0,0,0,1,0,1,0,1,1,1,1,0,1,1,1,0,0,0,1,0,1,1,
         0,0,0,1,0,1,1,1,1,0,1,1,1,0,1,0,0,0]  # Random binary string with 500 bits


# Set impulse parameters for binary field...
f_strength = 0.002
pulse_length = 0.01e-12  # [s]
pulse_interval = 0.15e-9  # [s]

# ... and sinusoidal field
A_ratio = 0.5  # []
A_hi = 0.0002  # [T]
frequency = 1  # [Hz]


# Create binary field...
datastream = hotspice.io.RandomBinaryDatastream(p0=0)  # Field is uniform over lattice
fi0 = hotspice.io.FieldInputter(datastream=input, angle=np.pi/2 + np.pi, magnitude=0)  # pi/2 aligns field with long axis of magnets
fi1 = hotspice.io.FieldInputter(datastream=input, angle=np.pi/2, magnitude=1)

# ... and sinusoidal field
fs0 = hotspice.io.FieldInputter(datastream=input, angle=np.pi/2, sine=True, frequency=frequency, magnitude=A_hi*A_ratio)
fs1 = hotspice.io.FieldInputter(datastream=input, angle=np.pi/2, sine=True, frequency=frequency, magnitude=A_hi)


# Set computation parameters
N = 50  # Num. input bits
R = 5  # Num. ASI runs per input
D = 15  # Num. data points per input to use as readout
mem = 1  # Num. bits into past to assess memory


def run_binary_input(mm: hotspice.Magnets, data=None):
    """
    This uses manual recalling of input_zero for each input bit.
    Applies field of N input bits to ASI, where each impulse length and interval is pre-set, and returns:
        data = [time [s], m_avg [Am^2], field status]
    Binary input string produces binary field sequence, where [1] turns field on and [0] keeps it off.
    """
    # Initialise data
    if input[0] == 1:   data = np.array([[mm.t, mm.m_avg, 1],])
    elif input[0] == 0: data = np.array([[mm.t, mm.m_avg, -1],])

    # Run through field impulses
    for f in range(N):
        # Apply binary field
        if input[f] == 1:
            fi1.input_zero(mm, value=f_strength)
        elif input[f] == 0:
            fi0.input_zero(mm, value=f_strength)

        # Let ASI adjust to field
        time_start = mm.t
        while mm.t - time_start <= pulse_length:
            mm.update()
            if input[f] == 1:
                data = np.append(data, [[mm.t, mm.m_avg, 1], ], axis=0)
            elif input[f] == 0:
                data = np.append(data, [[mm.t, mm.m_avg, -1], ], axis=0)
        #hotspice.plottools.show_m(mm)

        # Remove field
        fi1.input_zero(mm, value=0)

        # Run N switches
        time_start = mm.t
        while mm.t - time_start <= pulse_interval:
            mm.update()
            data = np.append(data, [[mm.t,mm.m_avg,0],],axis=0)
        #hotspice.plottools.show_m(mm)

    return data


def run_sine_input(mm: hotspice.Magnets, data=None):
    """
    This uses manual recalling of input_sine_data for each input bit.
    Applies field of N input bits to ASI, where input frequency is pre-set, and returns:
        data = [time [s], m_avg [Am^2], field magnitude]
    Binary input string produces alternating sinusoidal field, where [1] has greater amplitude than [0].
    """
    # Initialise data
    data = np.array([[mm.t, mm.m_avg, 0], ])

    # Run through field impulses
    for f in range(N):
        # Apply sinusoidal field
        if input[f] == 1:
            data = fs1.input_sine_data(mm, value=1, data=data)
        elif input[f] == 0:
            data = fs0.input_sine_data(mm, value=1, data=data)

    return data


def plot_response(data=None):
    """Takes data for one run and plots average magnetisation [Am^2] against time [s]."""
    fig,ax = plt.subplots(2,1)
    for i in range(2):
        ax[i].plot(data[:, 0], data[:, i + 1])
        ax[i].set_xlabel("Time /s")
    ax[0].set_ylabel("Average magnetisation /Am^2")
    ax[1].set_ylabel("Field magnitude")
    fig.suptitle(f"Average magnetisation over time as field encoding binary string is applied.")
    plt.show()


def append_readout(data=None, readout=None, run=None):
    """
    Takes existing readout and data for current run.
    Creates data_readout, an array((N,D)) of the first D data points (of data[1] = m_avg) for each of N input bits.
    Appends data_readout to existing readout and returns readout.
    """
    # Create readout layer for current run
    data_readout = np.array([])
    intervals = 0
    inputs = 1
    for i in range(data.shape[0]):
        data_readout_bit = np.array([])

        if not data[i][2] == 0:  # If field applied
            if intervals == inputs - 1:  # And it has just been applied

                # Add next D data points horizontally to data_readout_bit
                for p in range(D):
                    data_readout_bit = np.append(data_readout_bit, values=[data[i+p][1], ])

                # Append data_readout_bit to data_readout vertically
                if inputs == 1:
                    data_readout = np.array([data_readout_bit,])
                else:
                    data_readout = np.append(data_readout, np.array([data_readout_bit,]), axis=0)
                inputs += 1

        elif data[i][2] == 0: # If field not applied
            if intervals == inputs - 2:  # And it has just been removed
                intervals += 1

    # Append readout for current run to overall readout
    if run == 0:
        readout = data_readout
    else:
        readout = np.append(readout, data_readout, axis=1)

    return readout


def memory_task(input=None, mem=None):
    """Finds the string of bits corresponding to the input int(mem) bits in the past."""
    output = np.zeros(N)
    for n in range(N):
        output[n] = input[n-mem]
    return output


def linear_reg(readout=None, output=None, plot=False):
    """
    Applies ridge regression on the readout layer to adjust weights so that it matches the desired output.
    Splits the input string into training and test data in the ratio 2:1.
    Plots trained output against training output and predicted output against test output.
    Returns the mean squared error (MSE) in the ASI computing scheme's prediction.
    """
    # Proportions of train vs test data in output
    train_start = 0
    trainsize = int(N*2/3)
    testsize = int(N*1/3)

    # Split readout and output into train vs test data
    Xtr, Xte, Ytr, Yte = (readout[train_start:train_start + trainsize],
                          readout[train_start + trainsize:train_start + trainsize + testsize],
                          output[train_start:train_start + trainsize],
                          output[train_start + trainsize:train_start + trainsize + testsize])

    # Define the model
    model = Ridge()

    # Fit the training data
    model.fit(Xtr, Ytr)

    # Predict the train and test sets
    Ztr = model.predict(Xtr)
    Zte = model.predict(Xte)


    if plot == True:
        # Plot the weighted readout against train data
        plt.plot(Ytr, label='Training data (output)')
        plt.plot(Ztr, label='Trained prediction (weighted readout)')
        plt.legend()
        plt.show()

        # Plot the weighted readout against test data
        plt.plot(Yte, label='Test data (output)')
        plt.plot(Zte, label='Trained prediction (weighted readout)')
        plt.legend()
        plt.show()

    # Evaluate success
    error = mean_squared_error(Yte, Zte)
    print(f"Mean squared error = {error:.4f}")

    return error


def experiment(mem=None):
    """Tests the computational ability of an ASI to 'remember' a given number of input bits into the past."""
    data = np.zeros(3)  # Response from one run of input
    readout = []

    mm = hotspice.ASI.IP_Ising_Offset(a=a, b=b, n=n, T=T, E_B=E_B, angle=angle, pattern=pattern,
                                            energies=energies, Msat=Msat, V=V, PBC=True)  # Create ASI input

    for r in range(R):
        if binary: data = run_binary_input(mm, data=data)
        elif not binary: data = run_sine_input(mm, data=data)
        if r==0: plot_response(data=data)
        readout = append_readout(data=data, readout=readout, run=r)
        print(f"{r+1} runs done!")

    output = memory_task(input=input, mem=mem)
    error = linear_reg(readout=readout, output=output, plot=True)

    return error


def errors_sweep(error=None, errors=None, mem=None):
    """
    Appends the latest MSE to an array of MSEs for each memory task, and returns:
        errors = [mem, error]
    """
    if np.ndim(errors) == 1:    errors = np.array([[mem,error],])
    else:   errors = np.append(errors, np.array([[mem, error],]), axis=0)

    return errors


def experiment_sweep():
    """
    By running multiple experiments, tests the computational ability of the ASI for a range of different memory tasks.
    Tasks correspond to 'remembering' 1,2,3, and 4 input bits into the past.
    """
    errors = np.array([])  # Initialise MSE values
    mem_range = np.array([1,2,3,4])
    for m in range(mem_range.size):
        print(f"mem = {mem_range[m]}")
        for k in range(1):
            error = experiment(mem=mem_range[m])
            errors = errors_sweep(mem=mem_range[m], error=error, errors=errors)
    print(errors)



if __name__ == "__main__":
    experiment(mem=2)

