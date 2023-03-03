import os
import numpy as np
from gnss_ins_sim.sim import ins_sim
from gnss_ins_sim.sim import sim_data

fs = 100.0          # IMU sample frequency
fs_gps = 10.0       # GPS sample frequency
fs_mag = fs         # magnetometer sample frequency, not used for now

def plot_positions(data_dir):
    # start simulation to read data from files
    sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                      data_dir,
                      ref_frame=0,
                      imu=None,
                      mode=None,
                      env=None,
                      algorithm=None)
    sim.run(1)

    positions = sim_data.Sim_data(name='positions',\
                                  description='map street, fitted line, ins trajactory',\
                                  units=['deg', 'deg', 'm'],\
                                  output_units=['deg', 'deg', 'm'],\
                                  legend=['lat', 'lon', 'alt'])

    map_street = np.genfromtxt(".//tool_data//src//map_street.csv", delimiter=',', skip_header=1)
    map_street = map_street[:, [1, 0, 2]] # swap 0 and 1 columns since the input is [lon, lat, alt]
    positions.add_data(map_street, 'map_street', ['deg', 'deg', 'm'])

    fitted_line = np.genfromtxt(".//tool_data//src//fitted_line.csv", delimiter=',', skip_header=1)
    fitted_line = fitted_line[:, [1, 0, 2]]
    positions.add_data(fitted_line, 'fitted_line', ['deg', 'deg', 'm'])

    positions.add_data(sim.dmgr.ref_pos.data, 'ins_trajactory', ['rad', 'rad', 'm'])
    #positions.add_data(sim.dmgr.gps.data[0][:, 0:3], 'ins_trajactory_err', ['rad', 'rad', 'm'])

    positions.plot(sim.dmgr.time, plot3d=3)
    sim_data.show_plot()

if __name__ == '__main__':
    data_dir = os.path.abspath('.//demo_saved_data//')
    plot_positions(data_dir)
