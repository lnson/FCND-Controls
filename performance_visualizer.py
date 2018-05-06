import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

DEFAULT_PERFORMANCE_FILENAME = "control-result.pdf"

def PlotComparativeTwoDimensionalData(actual_x, actual_y,
                                      desired_x, desired_y,
                                      x_label, y_label):
    plt.plot(desired_x, desired_y, color='green')
    plt.plot(actual_x, actual_y, color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

def PlotComparativeOneDimensionalData(actual_data, desired_data, data_label):
    PlotComparativeTwoDimensionalData(range(len(actual_data)), actual_data,
                                      range(len(desired_data)), desired_data,
                                      'time', data_label)

def PlotOneDimensionalData(data, data_label):
    plt.plot(range(len(data)), data, color='red')
    plt.xlabel('time')
    plt.ylabel(data_label)
    plt.grid(True)

def PlotOneDimensionalDataWithThreshold(data, data_label, threshold):
    plt.plot(range(len(data)), data, color='red')
    good_part = data.copy()
    good_part[good_part > threshold] = np.nan
    plt.plot(range(len(good_part)), good_part, color='blue')
    threshold_line = [threshold] * len(good_part)
    plt.plot(range(len(threshold_line)), threshold_line, color='blue', linestyle='--')
    plt.xlabel('time')
    plt.ylabel(data_label)
    plt.grid(True)

def SavePerformanceReport(desired_trajectory, actual_trajectory, filename=DEFAULT_PERFORMANCE_FILENAME):
    plt.figure(figsize=(10, 16), dpi=1200)

    plt.subplot(4, 2, 1)
    PlotComparativeOneDimensionalData(-np.ravel(actual_trajectory[:, 2]),
                                      -np.ravel(desired_trajectory[:, 2]),
                                      '$z$')

    # z error
    plt.subplot(4, 2, 2)
    z_error = np.abs(np.ravel(desired_trajectory[:, 2]) -
                     np.ravel(actual_trajectory[:, 2]))
    PlotOneDimensionalDataWithThreshold(z_error, '$error_z$', 1.0)

    # xy trajectory
    plt.subplot(4, 2, 3)
    PlotComparativeTwoDimensionalData(np.ravel(actual_trajectory[:, 0]),
                                      np.ravel(actual_trajectory[:, 1]),
                                      np.ravel(desired_trajectory[:, 0]),
                                      np.ravel(desired_trajectory[:, 1]),
                                      '$x$', '$y$')

    # xy error
    plt.subplot(4, 2, 4)
    xy_error = np.linalg.norm(actual_trajectory[:, 0:2] -
                              desired_trajectory[:, 0:2], axis=1)
    PlotOneDimensionalDataWithThreshold(xy_error, 'Horizontal error', 2.0)

    # x trajectory
    plt.subplot(4, 2, 5)
    PlotComparativeOneDimensionalData(np.ravel(actual_trajectory[:, 0]),
                                      np.ravel(desired_trajectory[:, 0]),
                                      '$x$')

    # x error
    plt.subplot(4, 2, 6)
    x_error = np.abs(np.ravel(desired_trajectory[:, 0]) -
                     np.ravel(actual_trajectory[:, 0]))
    PlotOneDimensionalData(x_error, '$error_x$')

    # y trajectory
    plt.subplot(4, 2, 7)
    PlotComparativeOneDimensionalData(np.ravel(actual_trajectory[:, 1]),
                                      np.ravel(desired_trajectory[:, 1]),
                                      '$y$')

    # y_error
    plt.subplot(4, 2, 8)
    y_error = np.abs(np.ravel(desired_trajectory[:, 1]) -
                     np.ravel(actual_trajectory[:, 1]))
    PlotOneDimensionalData(y_error, '$error_y$')

    plt.savefig(filename)