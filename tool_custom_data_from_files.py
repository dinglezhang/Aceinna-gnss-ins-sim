"""
Tool to generate custom data from files.
Created on 2023-02-21
@author: dinglezhang
"""

import os
import numpy as np
from gnss_ins_sim.sim import ins_sim
from gnss_ins_sim.sim import sim_data

motion_def_path = os.path.abspath('.//demo_motion_def_files//')
fs = 100.0          # IMU sample frequency
fs_gps = 10.0       # GPS sample frequency
fs_mag = fs         # magnetometer sample frequency, not used for now

def gen_custom_data_from_files(data_dir):
    #### start simulation by reading data from files
    sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                      data_dir,
                      ref_frame=0,
                      imu=None,
                      mode=None,
                      env=None,
                      algorithm=None)
    sim.run(1)

    # all data name that need to write to one file
    __all_in_one = [
        sim.dmgr.ref_pos,
        sim.dmgr.ref_att_euler,
        sim.dmgr.ref_vel,
        sim.dmgr.ref_accel,
        sim.dmgr.ref_gyro,
        sim.dmgr.ref_odo
    ]

    all_in_one_data = sim_data.Sim_data(name='imu_custom_in_one',\
                                        description='all data output',\
                                        units=[],\
                                        output_units = [],\
                                        legend=[])

    for data in __all_in_one:
        if data.name in sim.dmgr.available:
            all_in_one_data.units.extend(data.units)
            all_in_one_data.output_units.extend(data.output_units)
            all_in_one_data.legend.extend(data.legend)
            if (len(all_in_one_data.data) == 0):  # first time to combine data
                all_in_one_data.data = data.data
            else:
              if (data.name == 'ref_odo'):
                  ref_odo_2d = np.atleast_2d(data.data)
                  ref_odo_2d = np.transpose(ref_odo_2d)
                  all_in_one_data.data = np.append(all_in_one_data.data, ref_odo_2d, 1)
              else:
                  all_in_one_data.data = np.append(all_in_one_data.data, data.data, 1)

    # ref_Yaw, ref_Pitch, ref_Roll -> ref_Roll, ref_Yaw, ref_Pitch
    all_in_one_data.legend[3] = 'ref_Pitch'
    all_in_one_data.legend[4] = 'ref_Roll'
    all_in_one_data.legend[5] = 'ref_Yaw'
    tmp_column = all_in_one_data.data[:, 3]
    tmp_array = np.delete(all_in_one_data.data, 3, 1)
    all_in_one_data.data = np.insert(tmp_array, 5, tmp_column, 1)

    # ref_vel from NED to ENU
    tmp_column = all_in_one_data.data[:, 6]
    tmp_array = np.delete(all_in_one_data.data, 6, 1)
    all_in_one_data.data = np.insert(tmp_array, 7, tmp_column, 1)
    all_in_one_data.data[:, 8] = -all_in_one_data.data[:, 8]

    # ref_accel from NED to ENU
    tmp_column = all_in_one_data.data[:, 9]
    tmp_array = np.delete(all_in_one_data.data, 9, 1)
    all_in_one_data.data = np.insert(tmp_array, 10, tmp_column, 1)
    all_in_one_data.data[:, 11] = -all_in_one_data.data[:, 11]

    # ref_gyro from NED to ENU
    tmp_column = all_in_one_data.data[:, 12]
    tmp_array = np.delete(all_in_one_data.data, 12, 1)
    all_in_one_data.data = np.insert(tmp_array, 13, tmp_column, 1)
    all_in_one_data.data[:, 14] = -all_in_one_data.data[:, 14]

    # ref_odo to ref_dist
    all_in_one_data.legend[15] = 'ref_dist'
    all_in_one_data.units[15] = 'm'
    all_in_one_data.output_units[15] = 'm'
    all_in_one_data.data[:, 15] = all_in_one_data.data[:, 15]/100

    # 0~2 is ref_pos, no need to change from deg to rad for them
    for i in range(3, len(all_in_one_data.output_units)):
      if (all_in_one_data.output_units[i]) == 'deg':
        all_in_one_data.output_units[i] = 'rad'
      if (all_in_one_data.output_units[i]) == 'deg/s':
        all_in_one_data.output_units[i] = 'rad/s'

    print('saving %s'% all_in_one_data.name)
    all_in_one_data.save_to_file(data_dir)

if __name__ == '__main__':
    dir_of_logged_files = os.path.abspath('.//demo_saved_data//')
    gen_custom_data_from_files(dir_of_logged_files)
