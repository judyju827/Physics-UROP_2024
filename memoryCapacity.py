import os
os.environ['HOTSPICE_USE_GPU'] = 'False'

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

try: from context import hotspice
except ModuleNotFoundError: import hotspice

geometry = 'Ising_Offset'

# Set pinwheel parameters
if geometry == 'Pinwheel':
    a = 600e-9
    θ = 70
    b = a * np.tan(θ * np.pi / 180)
    n = 21
    T = 300
    E_B = hotspice.utils.eV_to_J(71)
    moment = 860e3 * 470e-9 * 170e-9 * 10e-9
    ext_field = 0.07
    ext_angle = 7 * math.pi / 180
    sine = True
    frequency = 1
    res_x = 5
    res_y = 5


# Set offset Ising parameters
if geometry == 'Ising_Offset':
    a = 280e-9
    θ = 70
    b = a * np.tan(θ * np.pi / 180)
    n = 20
    T = 300
    E_B = 5e-22
    moment = 800e3 * 4.8e-23
    ext_field = 0.00015
    ext_angle = np.pi/2
    sine = True
    frequency = 1
    res_x = 5
    res_y = 5


# Test memory capacity of single ASI
def test_MC():
    mm = hotspice.ASI.IP_Ising_Offset(a=a, b=b, angle=np.pi/2, n=n, T=T, E_B=E_B, moment=moment, PBC=False, pattern='random',
                                  energies=(hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()))

    datastream = hotspice.io.RandomBinaryDatastream()
    inputter = hotspice.io.PerpFieldInputter(datastream, magnitude=ext_field, angle=ext_angle, n=2,
                                             sine=sine, frequency=frequency)
    outputreader = hotspice.io.RegionalOutputReader(res_x, res_y, mm)
    experiment = hotspice.experiments.TaskAgnosticExperiment(inputter, outputreader, mm)

    experiment.run(N=1000, pattern='random', verbose=1)
    experiment.calculate_all()
    NL = experiment.NL(use_stored=True)
    MC = experiment.MC()
    print(NL, MC)


# Apply linear regression to ASI output
def compute_memory(mem=1):
    # Apply field to ASI
    mm = hotspice.ASI.IP_Ising_Offset(a=a, b=b, angle=np.pi / 2, n=n, T=T, E_B=E_B, moment=moment, PBC=False,
                                      pattern='random',
                                      energies=(hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()))

    datastream = hotspice.io.RandomBinaryDatastream()
    inputter = hotspice.io.PerpFieldInputter(datastream, magnitude=ext_field, angle=ext_angle, n=2,
                                             sine=sine, frequency=frequency)
    outputreader = hotspice.io.RegionalOutputReader(res_x, res_y, mm)
    experiment = hotspice.experiments.TaskAgnosticExperiment(inputter, outputreader, mm)

    experiment.run(N=1000, pattern='random', verbose=1)

    # Create signal & target arrays
    signal = experiment.u[:]
    target = np.zeros(len(signal))
    for i in range(len(signal)):
        target[i] = signal[i-mem]

    # Create readout array
    ## TODO: plot readout as function of signal, check what it represents & probably adjust
    readout = experiment.y[:,:]
    print(readout.shape)

    # Apply ridge regression to weights
    train_start = 0
    trainsize = int(len(signal)*2/3)
    testsize = int(len(signal)*1/3)
    Xtr, Xte, Ytr, Yte = (readout[train_start:train_start + trainsize],
                          readout[train_start + trainsize:train_start + trainsize + testsize],
                          target[train_start:train_start + trainsize],
                          target[train_start + trainsize:train_start + trainsize + testsize])
    model=Ridge()
    model.fit(Xtr, Ytr)
    Ztr, Zte = (model.predict(Xtr), model.predict(Xte))

    # Plot training data
    plt.plot(Ytr, label='Training data (output)')
    plt.plot(Ztr, label='Trained prediction (weighted readout)')
    plt.legend()
    plt.show()

    # Plot test data
    plt.plot(Yte, label='Test data (output)')
    plt.plot(Zte, label='Trained prediction (weighted readout)')
    plt.legend()
    plt.show()

    # Evaluate success
    error = mean_squared_error(Yte, Zte)
    print(f"Mean squared error = {error:.4f}")


if __name__ == "__main__":
    compute_memory()