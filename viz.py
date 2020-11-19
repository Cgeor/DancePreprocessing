"""
    Functions to visualize human poses
    adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import json
from scipy.interpolate import make_lsq_spline, BSpline, LSQUnivariateSpline

'''Visualization script for the output file from pose_optim'''

class Ax3DPose(object):
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c", label=['P1', 'P2']):
        """
        Create a 3d pose visualizer that can be updated with new poses.
        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I = np.array([0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 11, 11, 22, 8, 12, 13, 14, 14, 19, 0, 15, 0, 16])
        self.J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 24, 22, 23, 12, 13, 14, 21, 19, 20, 15, 17, 16, 18])
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
        self.ax = ax

        vals = np.zeros((25, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots.append(
                    self.ax.plot(x, z, y, lw=2, linestyle='--', c=rcolor if self.LR[i] else lcolor, label=label[0]))
            else:
                self.plots.append(self.ax.plot(x, y, z, lw=2, linestyle='--', c=rcolor if self.LR[i] else lcolor))

        self.plots_pred = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c=rcolor if self.LR[i] else lcolor, label=label[1]))
            else:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c=rcolor if self.LR[i] else lcolor))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # self.ax.set_axis_off()
        # self.ax.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        self.ax.legend(loc='lower left')
        self.ax.view_init(130, -90)

        r = 1000
        self.ax.set_xlim3d([-r, r])
        self.ax.set_zlim3d([-r, r])
        self.ax.set_ylim3d([-0.5 * r, 2 * r])
        self.ax.set_aspect('auto')

    def update(self, gt_channels, pred_channels):
        """
        Update the plotted 3d pose.
        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert gt_channels.size == 75, "channels should have 96 entries, it has %d instead" % gt_channels.size
        gt_vals = np.reshape(gt_channels, (25, -1))
        lcolor = "#8e8e8e"
        rcolor = "#383838"
        for i in np.arange(len(self.I)):
            x = np.array([gt_vals[self.I[i], 0], gt_vals[self.J[i], 0]])
            y = np.array([gt_vals[self.I[i], 1], gt_vals[self.J[i], 1]])
            z = np.array([gt_vals[self.I[i], 2], gt_vals[self.J[i], 2]])
            if (not np.any(gt_vals[self.I[i]] == np.inf)) and (not np.any(gt_vals[self.J[i]] == np.inf)):
                self.plots[i][0].set_xdata(x)
                self.plots[i][0].set_ydata(y)
                self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots[i][0].set_alpha(0.5)

        assert pred_channels.size == 75, "channels should have 96 entries, it has %d instead" % pred_channels.size
        pred_vals = np.reshape(pred_channels, (25, -1))
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
        for i in np.arange(len(self.I)):
            x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
            y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
            z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])
            if (not pred_vals[self.I[i], 0] == np.inf) and (not pred_vals[self.J[i], 0] == np.inf):
                self.plots_pred[i][0].set_xdata(x)
                self.plots_pred[i][0].set_ydata(y)
                self.plots_pred[i][0].set_3d_properties(z)
            self.plots_pred[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots_pred[i][0].set_alpha(0.7)



def plot_predictions_from_3d(gt, pred, fig, ax, f_title):

    nframes_pred = pred.shape[0]
    # === Plot and animate ===
    ob = Ax3DPose(ax)

    # Plot the prediction
    for i in range(nframes_pred):

        ob.update(gt[i, :], pred[i, :])
        ax.set_title(f_title + ' frame:{:d}'.format(i + 1), loc="left")
        plt.show(block=False)

        fig.canvas.draw()
        filename = 'images/img_step_' + str(i) + '.png'
        plt.savefig(filename, dpi=96)
        plt.pause(0.05)


def to_mat(data):
    n_frames = len(data.keys())
    mat = np.full((n_frames, 25, 3), np.inf)

    for frame in range(n_frames):
        xyz = data[f"frame{frame}"]["points_3d"]
        ids = data[f"frame{frame}"]["ids"]
        for i, id in enumerate(ids):
            mat[frame][id] = xyz[i]

    return mat

fig = plt.figure()
ax = plt.gca(projection='3d')

with open('seq7_poses_4.json') as file:
    dict = json.load(file)

with open('all_frames_sub1.json') as file:
    data1 = json.load(file)

with open('all_frames_sub2.json') as file:
    data2 = json.load(file)

mat1 = dict['p1']
mat2 = dict['p2']
mat1 = np.array(mat1)
mat2 = np.array(mat2)
'''
mat1_init = to_mat(data1)
mat2_init = to_mat(data2)

y1 = mat2[:, 0, 0]
y2 = mat2_init[:, 0, 0]
x = np.arange(len(y1))
t = list(x[::40])
t = t[1:]
w = np.isnan(y1)
w2 = np.isinf(y1)
y1[w] = 0
print(y1)
cs = LSQUnivariateSpline(x, y1, t,  w=~w2)
plt.plot(x[0:1000], y1[0:1000])
plt.plot(x[0:1000], y2[0:1000])
plt.show()
'''

plot_predictions_from_3d(mat1, mat2, fig, ax, 'title')