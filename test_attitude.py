import math
import numpy as np
from gnss_ins_sim.attitude import attitude
from scipy.spatial.transform import Rotation

PASS_STR = "\033[1;32mPASS\033[0m"       # with green color
FAILED_STR = "\033[1;31mFAILED\033[0m"   # with red color

def test_euler_2_x():
    euler_d_input = np.array([20, 80, 15])
    euler_r_input = euler_d_input * attitude.D2R
    print('***euler input by zyx seqence:')
    print('%s (%s)\n'% (euler_d_input, euler_r_input))

    dcm = attitude.euler2dcm(euler_r_input, 'zyx')
    rot = Rotation.from_euler('ZYX', euler_r_input) # in body frame (intrinsic)

    dcm_scipy = rot.as_matrix()
    result = FAILED_STR
    if np.allclose(dcm, dcm_scipy.T): # attitude is to rotate frame for postmultiplication, but scipy is to rotate vector
        result =  PASS_STR
    print('***dcm output: %s'% result)
    print('attitude:\n%s'% dcm)
    print('scipy: \n%s\n'% dcm_scipy)

    quat = attitude.dcm2quat(dcm)
    quat_scipy = rot.as_quat()
    result = FAILED_STR
    if np.allclose(quat, quat_scipy[[3, 0, 1, 2]]):
        result =  PASS_STR
    print('***quat output: %s'% result)
    print('attitude:\n%s'% quat)
    print('scipy: \n%s\n'% quat_scipy)

    for seq in (\
        # three axis rotation
        'zyx', 'zxy', 'yxz', 'yzx', 'xyz', 'xzy', \
        # two axis rotation
        'zyz', 'zxz', 'yxy', 'yzy', 'xyx', 'xzx'\
        ):
        euler_r_output = attitude.dcm2euler(dcm, seq)
        euler_d_output = euler_r_output * attitude.R2D

        euler_r_output_scipy = rot.as_euler(seq.upper())
        euler_d_output_scipy = euler_r_output_scipy * attitude.R2D

        result = FAILED_STR
        if np.allclose(euler_r_output, euler_r_output_scipy):
            result =  PASS_STR

        print('***euler output by %s sequence: %s'% (seq, result))
        print('attitude: %s (%s)'% (euler_d_output, euler_r_output))
        print('scipy:    %s (%s)\n'% (euler_d_output_scipy, euler_r_output_scipy))

if __name__ == '__main__':
    test_euler_2_x()
