import math
import numpy as np
from gnss_ins_sim.attitude import attitude
from scipy.spatial.transform import Rotation

def test_euler_2_x():
    euler_d_input = np.array([45, 45, 45])
    euler_r_input = euler_d_input * attitude.D2R
    print('euler input by zyx seqence:')
    print('%s (%s)\n'% (euler_d_input, euler_r_input))

    dcm = attitude.euler2dcm(euler_r_input, 'zyx')
    rot = Rotation.from_euler('zyx', euler_r_input)

    for seq in ('zyx', 'zxy', 'yxz', 'yzx', 'xyz', 'xzy'):
        euler_r_output = attitude.dcm2euler(dcm, seq)
        euler_d_output = euler_r_output * attitude.R2D

        euler_r_output_scipy = rot.as_euler(seq)
        euler_d_output_scipy = euler_r_output_scipy * attitude.R2D

        result = 'FAILED'
        if np.allclose(euler_r_output, euler_r_output_scipy):
            result =  'PASS'

        print('euler output by %s sequence: %s'% (seq, result))
        print('attitude: %s (%s)'% (euler_d_output, euler_r_output))
        print('scipy:    %s (%s)\n'% (euler_d_output_scipy, euler_r_output_scipy))

if __name__ == '__main__':
    test_euler_2_x()
