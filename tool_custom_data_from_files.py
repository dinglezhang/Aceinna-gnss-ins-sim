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

    description_en = \
'''1~3: true LLA pos
4~6: true attitude (Euler angles, XYZ rotation sequency) in the ENU frame
7~9: true vel in the ENU frame
10~12: true acceleration in the body RFU frame
13~15: true angular velocity (XYZ rotation sequency) in the body RFU frame
16: true distance per 10ms
'''
    description_zh = \
'''各列含义介绍：
1~3: 经度，纬度，高度
4~6: 俯仰角，横滚角，航向角（XYZ旋转顺序，“东北天”坐标系）
7~9: 东向速度，北向速度，天向速度（“东北天”坐标系）
10~12: 车身右侧加速度，车身前方加速度，车身上方加速度（车身“右前上”坐标系）
13~15: 俯仰角速度，横滚角速度，航向角速度（XYZ旋转顺序，车身“右前上”坐标系）
16: 车前方每10ms里程
以上均为理想值。
'''

    output_units = [
        'deg', 'deg', 'm',

        'rad', 'rad', 'rad',
        #'deg', 'deg', 'deg',

        'm/s', 'm/s', 'm/s',

        'm/s^2', 'm/s^2', 'm/s^2',

        'rad/s', 'rad/s', 'rad/s',
        #'deg/s', 'deg/s', 'deg/s',

        'm'
    ]
    legend = [
        'lon', 'lat', 'alt',
        'pitch', 'roll', 'yaw',
        'vel_x', 'vel_y', 'vel_z',
        'accel_x', 'accel_y', 'accel_z',
        'pitch_vel', 'roll_vel', 'yaw_vel',
        'dist'
    ]
    custom_data = sim_data.Sim_data(name='imu_all',\
                                    description=description_zh,\
                                    units=[],\
                                    output_units=output_units,\
                                    legend=legend)
    custom_data.units = []  # set units back since it is set to output_units by Sim_data

    # all data that need to tranform and write to one file
    all_data_needed = [
        sim.dmgr.ref_pos,
        sim.dmgr.ref_att_euler,
        sim.dmgr.ref_vel,
        sim.dmgr.ref_accel,
        sim.dmgr.ref_gyro,
        sim.dmgr.ref_odo
    ]
    row_num = -1
    for data in all_data_needed:
        if data.name in sim.dmgr.available:
            custom_data.units.extend(data.units)
            cur_row_num = data.data.shape[0]
            if (row_num == -1):
                row_num = cur_row_num
            else:
                if (row_num != cur_row_num):
                    raise ValueError('Row number of %s data %d is wrong, should be %d'% (data.name, cur_row_num, row_num))
        else:
            raise ValueError('%s data is missing'% data.name)

    custom_data.data = np.zeros((row_num, 16))

    # lat, lon, alt -> lon, lat, alt
    custom_data.data[:, 0] = sim.dmgr.ref_pos.data[:, 1]
    custom_data.data[:, 1] = sim.dmgr.ref_pos.data[:, 0]
    custom_data.data[:, 2] = sim.dmgr.ref_pos.data[:, 2]

    # ref_att_euler from NED(yaw, pitch, roll) to ENU(pitch, roll, yaw)
    custom_data.data[:, 3:6] = sim.array_quat2euler(sim.dmgr.ref_att_quat.data, 'yxz')
    custom_data.data[:, 5] = -custom_data.data[:, 5]

    # ref_vel from NED to ENU
    custom_data.data[:, 6] = sim.dmgr.ref_vel.data[:, 1]
    custom_data.data[:, 7] = sim.dmgr.ref_vel.data[:, 0]
    custom_data.data[:, 8] = -sim.dmgr.ref_vel.data[:, 2]

    # ref_accel from FRD to RFU
    custom_data.data[:, 9] = sim.dmgr.ref_accel.data[:, 1]
    custom_data.data[:, 10] = sim.dmgr.ref_accel.data[:, 0]
    custom_data.data[:, 11] = -sim.dmgr.ref_accel.data[:, 2]

    # ref_gyro from FRD(yaw, pitch, roll) to RFU(pitch, roll, yaw)
    ref_gyro_quat = sim.array_euler2quat(sim.dmgr.ref_gyro.data)
    custom_data.data[:, 12:15] = sim.array_quat2euler(ref_gyro_quat, 'yxz')
    custom_data.data[:, 14] = -custom_data.data[:, 14]

    # ref_odo to ref_dist
    custom_data.units[15] = 'm'
    custom_data.data[:, 15] = sim.dmgr.ref_odo.data.reshape((row_num, 1))[:, 0]/fs

    # save custom date file and readme
    print('saving %s'% custom_data.name)
    custom_data.save_to_file(data_dir)

    read_me_name = data_dir + '//' + custom_data.name + '_readme.txt'
    with open(read_me_name, 'w', encoding='UTF-8') as fp:
        fp.write(custom_data.description)

if __name__ == '__main__':
    data_dir = os.path.abspath('.//demo_saved_data//')
    gen_custom_data_from_files(data_dir)
