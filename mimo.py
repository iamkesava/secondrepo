# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:15:19 2023

@author: kesav
"""

import DeepMIMO
import numpy as np

# Load the default parameters
parameters = DeepMIMO.default_params()

# Set scenario name
parameters['scenario'] = 'O1_drone_200'

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = r'E:\scenarios'


# Generate data
dataset = DeepMIMO.generate_data(parameters)

# If the O1_60 scenario is extracted in "C:/dataset/" folder, set
parameters['dataset_folder'] = r'C:/dataset/'

# The default value is set as './Raytracing_scenarios/'

# To load scenario O1_60, set the dictionary variable by
parameters['scenario'] = 'O1_60'

# To load the first five scenes, set
parameters['dynamic_settings']['first_scene'] = 1
parameters['dynamic_settings']['last_scene'] = 5

# To only include 10 strongest paths in the channel computation, set
parameters['num_paths'] = 10

fig = plt.figure()
ax = fig.add_subplot()
# Visualize channel magnitude response
# First, select indices of a user and bs
ue_idx = 0
bs_idx = 0
# Import channel
channel = dataset[bs_idx]['user']['channel'][ue_idx]
# Take only the first antenna pair
im = plt.imshow(np.abs(np.squeeze(channel).T))
plt.title('Channel Magnitude Response')
plt.xlabel('TX Antennas')
plt.ylabel('Subcarriers')
cbar = fig.colorbar(im, ax=ax)

loc_x = dataset[bs_idx]['user']['location'][:, 0]
loc_y = dataset[bs_idx]['user']['location'][:, 1]
loc_z = dataset[bs_idx]['user']['location'][:, 2]
pathloss = dataset[bs_idx]['user']['pathloss']
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
im = ax.scatter(loc_x, loc_y, loc_z, c=pathloss)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')

bs_loc_x = dataset[bs_idx]['basestation']['location'][:, 0]
bs_loc_y = dataset[bs_idx]['basestation']['location'][:, 1]
bs_loc_z = dataset[bs_idx]['basestation']['location'][:, 2]
ax.scatter(bs_loc_x, bs_loc_y, bs_loc_z, c='r')
ttl = plt.title('UE and BS Positions')
cbar = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.1)
cbar_title = cbar.ax.set_ylabel('UE Path-Loss (dBm)', rotation=270)
cbar.ax.get_yaxis().labelpad = 20

fig = plt.figure()
ax = fig.add_subplot()
im = ax.scatter(loc_x, loc_y, c=pathloss)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.colorbar(im, ax=ax)
ttl = plt.title('UE Grid Path-loss (dBm)')
