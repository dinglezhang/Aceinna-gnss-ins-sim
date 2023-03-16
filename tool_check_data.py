import os
import math
import numpy as np
from scipy.spatial.transform import Rotation

from gnss_ins_sim.sim import ins_sim
from gnss_ins_sim.sim import sim_data
from gnss_ins_sim.attitude import attitude
from tool_custom_data import convert_att_euler_frame_ned_2_enu

fs = 100.0          # IMU sample frequency
fs_gps = 10.0       # GPS sample frequency
fs_mag = fs         # magnetometer sample frequency, not used for now

def plot_positions(input_data_dir):
    # start simulation to read data from files
    sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                      input_data_dir,
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

def build_rotation_matrix(rotation_quaternion):
    """Construct a 3x3 rotation matrix given a rotation quaternion"""
    q0 = rotation_quaternion[0][0]
    q1 = rotation_quaternion[1][0]
    q2 = rotation_quaternion[2][0]
    q3 = rotation_quaternion[3][0]
    return np.array([
        [2*q0**2 + 2*q1**2 - 1, 2*(q1*q2 - q3*q0),      2*(q1*q3 + q2*q0)],
        [2*(q1*q2 + q3*q0),     2*q0**2 + 2*q2**2 - 1,  2*(q2*q3 - q1*q0)],
        [2*(q1*q3 - q2*q0),     2*(q2*q3 + q1*q0),      2*q0**2 + 2*q3**2 - 1]
    ])

def convert_att_euler_frame_ned_2_enu_by_quat(att_quat_ned_2_frd):
    ned_2_enu_quaternion = np.array([
        [0],
        [-math.sqrt(2)/2],
        [-math.sqrt(2)/2],
        [0]
        ])
    ned_2_enu_rotation_matrix = build_rotation_matrix(ned_2_enu_quaternion)

    frd_heading = np.array([[1], [0], [0]])
    rfu_heading = np.array([[0], [1], [0]])
    att_euler_enu_2_rfu = np.zeros([att_quat_ned_2_frd.shape[0], 3])
    for i in range(0, att_quat_ned_2_frd.shape[0]):
        vec_ned = attitude.quat2dcm(att_quat_ned_2_frd[i]).T.dot(frd_heading)
        vec_enu = ned_2_enu_rotation_matrix.dot(vec_ned)

        crossv = attitude.cross3(rfu_heading, vec_enu)
        crossv = crossv / np.linalg.norm(crossv)
        theta  = math.acos(vec_enu.reshape(3).dot(rfu_heading.reshape(3))/(np.linalg.norm(vec_enu) * np.linalg.norm(rfu_heading)))

        q = Rotation.from_rotvec(theta * crossv.reshape(3))
        att_euler_enu_2_rfu[i] = q.as_euler('ZYX')

    return att_euler_enu_2_rfu

def save_att_euler_to_file(att_euler_data, name, data, output_data_dir):
    att_euler_data.name = name
    att_euler_data.data = data

    print('saving %s'% att_euler_data.name)
    att_euler_data.save_to_file(output_data_dir, fmt='%.8f')

def att_euler_frame_ned_2_enu(input_data_dir, output_data_dir):
    # start simulation to read data from files
    sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                      input_data_dir,
                      ref_frame=0,
                      imu=None,
                      mode=None,
                      env=None,
                      algorithm=None)
    sim.run(1)

    att_euler_data = sim_data.Sim_data(name='att_euler',
                                       description='',
                                       units=['rad', 'rad', 'rad'],
                                       output_units=['rad', 'rad', 'rad'],
                                       legend=['pitch', 'roll', 'yaw'])

    # reuqirement is wired, but we follow them
    # ratation sequence is ZYX ([yaw, roll, pitch] in RFU), but output sequence is [pitch, roll, yaw]
    # anticlockwise is positive angle along z axis in RFU, but requirement asks that clockwise is positive

    # swap output sequence directly, but the rotation sequence is not correct
    att_euler_swapped = sim.dmgr.ref_att_euler.data[:, [1, 2, 0]]
    save_att_euler_to_file(att_euler_data, 'att_euler_swapped', att_euler_swapped, output_data_dir)

    # change rotation seqence, that looks a correct way for NED->ENU
    att_euler_zxy = sim.array_quat2euler(sim.dmgr.ref_att_quat.data, 'zxy')
    att_euler_zxy = att_euler_zxy[:, [2, 1, 0]]
    save_att_euler_to_file(att_euler_data, 'att_euler_zxy', att_euler_zxy, output_data_dir)

    # correct solution
    att_euler_enu_2_rfu = convert_att_euler_frame_ned_2_enu(sim.dmgr.ref_att_euler.data)
    att_euler_enu_2_rfu = att_euler_enu_2_rfu[:, [2, 1, 0]]
    att_euler_enu_2_rfu[:, 2] = -att_euler_enu_2_rfu[:, 2]
    save_att_euler_to_file(att_euler_data, 'att_euler_rotvec', att_euler_enu_2_rfu, output_data_dir)

    # calculate rotvec by two vectors, but it is not correct since it lacks of one dimension of self-rotation
    att_euler_enu_2_rfu_quat = convert_att_euler_frame_ned_2_enu_by_quat(sim.dmgr.ref_att_quat.data)
    att_euler_enu_2_rfu_quat = att_euler_enu_2_rfu_quat[:, [2, 1, 0]]
    att_euler_enu_2_rfu_quat[:, 2] = -att_euler_enu_2_rfu_quat[:, 2]
    save_att_euler_to_file(att_euler_data, 'att_euler_quat', att_euler_enu_2_rfu_quat, output_data_dir)

if __name__ == '__main__':
    input_data_dir = os.path.abspath('.//demo_saved_data//')
    output_data_dir = os.path.abspath('.//tool_data//output//')

    plot_positions(input_data_dir)

    att_euler_frame_ned_2_enu(input_data_dir, output_data_dir)
