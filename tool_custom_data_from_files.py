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
    # start simulation by reading data from files
    sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                      data_dir,
                      ref_frame=0,
                      imu=None,
                      mode=None,
                      env=None,
                      algorithm=None)
    sim.run(1)

    # all data that need to tranform and write to one file
    all_data_needed = [
        sim.dmgr.ref_pos,
        sim.dmgr.ref_att_euler,
        sim.dmgr.ref_vel,
        sim.dmgr.ref_accel,
        sim.dmgr.ref_gyro,
        sim.dmgr.ref_odo
    ]

    description = \
'''1~3: true LLA pos in the navigation frame
4~6: true attitude (Euler angles, XYZ)
7~9: true vel in the ENU frame
10~12: true accel in the body RFU frame
13~15: true angular velocity in the body RFU frame
16: true distance per 10ms
'''
    output_units = [
        'deg', 'deg', 'm',
        'rad', 'rad', 'rad',
        'm/s', 'm/s', 'm/s',
        'm/s^2', 'm/s^2', 'm/s^2',
        'rad/s', 'rad/s', 'rad/s',
        'm'
    ]
    legend = [
        'pos_lat', 'pos_lon', 'pos_alt',
        'Pitch', 'Roll', 'Yaw',
        'vel_x', 'vel_y', 'vel_z',
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'dist'
    ]
    custom_data = sim_data.Sim_data(name='imu_all',\
                                    description=description,\
                                    units=[],\
                                    output_units=output_units,\
                                    legend=legend)

    for data in all_data_needed:
        if data.name in sim.dmgr.available:
            custom_data.units.extend(data.units)
            if (len(custom_data.data) == 0):  # first time to combine data
                custom_data.data = data.data
            else:
              if (data.name == 'ref_odo'):
                  ref_odo_2d = np.atleast_2d(data.data)
                  ref_odo_2d = np.transpose(ref_odo_2d)
                  custom_data.data = np.append(custom_data.data, ref_odo_2d, 1)
              else:
                  custom_data.data = np.append(custom_data.data, data.data, 1)

    # ref_Yaw, ref_Pitch, ref_Roll -> ref_Roll, ref_Yaw, ref_Pitch
    tmp_column = custom_data.data[:, 3]
    tmp_array = np.delete(custom_data.data, 3, 1)
    custom_data.data = np.insert(tmp_array, 5, tmp_column, 1)

    # ref_vel from NED to ENU
    tmp_column = custom_data.data[:, 6]
    tmp_array = np.delete(custom_data.data, 6, 1)
    custom_data.data = np.insert(tmp_array, 7, tmp_column, 1)
    custom_data.data[:, 8] = -custom_data.data[:, 8]

    # ref_accel from NED to ENU
    tmp_column = custom_data.data[:, 9]
    tmp_array = np.delete(custom_data.data, 9, 1)
    custom_data.data = np.insert(tmp_array, 10, tmp_column, 1)
    custom_data.data[:, 11] = -custom_data.data[:, 11]

    # ref_gyro from NED to ENU
    tmp_column = custom_data.data[:, 12]
    tmp_array = np.delete(custom_data.data, 12, 1)
    custom_data.data = np.insert(tmp_array, 13, tmp_column, 1)
    custom_data.data[:, 14] = -custom_data.data[:, 14]

    # ref_odo to ref_dist
    custom_data.units[15] = 'm'
    custom_data.data[:, 15] = custom_data.data[:, 15]/100

    # save custom date file and readme
    print('saving %s'% custom_data.name)
    custom_data.save_to_file(data_dir)

    read_me_name = data_dir + '//' + custom_data.name + '_readme.txt'
    with open(read_me_name, 'w') as fp:
        fp.write(custom_data.description)

if __name__ == '__main__':
    data_dir = os.path.abspath('.//demo_saved_data//')
    gen_custom_data_from_files(data_dir)
