import math
import numpy as np
from gnss_ins_sim.attitude import attitude
from scipy.spatial.transform import Rotation

PASS_STR = "\033[1;32mPASS\033[0m"       # with green color
FAILED_STR = "\033[1;31mFAILED\033[0m"   # with red color

ROTATION_SEQUENCES = (\
    # three axis rotation
    'zyx', 'zxy', 'yxz', 'yzx', 'xyz', 'xzy', \
    # two axis rotation
    'zyz', 'zxz', 'yxy', 'yzy', 'xyx', 'xzx'\
)

# euler2dcm()
# euler2quat()
# dcm2euler() and quat2euler() by all rotation sequneces. (looks like euler to euler from input)
# dcm2quat() and quat2dcm() as well
def test_euler_2_x(euler_d_input, rot_seq):
    print('========================================================')

    euler_r_input = euler_d_input * attitude.D2R
    print('***euler input by %s seqence:***'% rot_seq)
    print('%s (%s)\n'% (euler_d_input, euler_r_input))

    rot = Rotation.from_euler(rot_seq.upper(), euler_r_input) # in body frame (intrinsic)

    # test euler2dcm()
    dcm = attitude.euler2dcm(euler_r_input, rot_seq)
    dcm_scipy = rot.as_matrix()
    result = FAILED_STR
    if np.allclose(dcm, dcm_scipy.T): # attitude is to rotate frame for postmultiplication, but scipy is to rotate vector
        result =  PASS_STR
    print('***dcm output: %s***'% result)
    print('attitude:\n%s'% dcm)
    print('scipy: \n%s\n'% dcm_scipy)

    # test euler2quat()
    quat = attitude.euler2quat(euler_r_input, rot_seq)
    quat_scipy = rot.as_quat()
    result = FAILED_STR
    if np.allclose(quat, quat_scipy[[3, 0, 1, 2]]):
        result =  PASS_STR
    print('***quat output: %s***'% result)
    print('attitude:\n%s'% quat)
    print('scipy: \n%s\n'% quat_scipy)

    # test dcm2quat() and quat2dcm() as well
    quat = attitude.dcm2quat(dcm)
    if not np.allclose(quat, quat_scipy[[3, 0, 1, 2]]):
        print('***dcm2quat() %s***'% FAILED_STR)
        return
    dcm = attitude.quat2dcm(quat)
    if not np.allclose(dcm, dcm_scipy.T):
        print('***quat2dcm() %s***'% FAILED_STR)
        return

    # test dcm2euler() and quat2euler() for all kinds of rotation sequences
    for seq in ROTATION_SEQUENCES:
        euler_r_output_by_dcm = attitude.dcm2euler(dcm, seq)
        euler_d_output_by_dcm = euler_r_output_by_dcm * attitude.R2D

        euler_r_output_by_quat = attitude.quat2euler(quat, seq)
        euler_d_output_by_quat = euler_r_output_by_dcm * attitude.R2D

        euler_r_output_scipy = rot.as_euler(seq.upper())
        euler_d_output_scipy = euler_r_output_scipy * attitude.R2D

        result = FAILED_STR
        if np.allclose(euler_r_output_by_dcm, euler_r_output_by_quat) and \
           np.allclose(euler_r_output_by_dcm, euler_r_output_scipy):
            result =  PASS_STR

        print('***euler output by %s sequence: %s***'% (seq, result))
        print('attitude by dcm:  %s (%s)'% (euler_d_output_by_dcm, euler_r_output_by_dcm))
        print('attitude by quat: %s (%s)'% (euler_d_output_by_quat, euler_r_output_by_quat))
        print('scipy:            %s (%s)\n'% (euler_d_output_scipy, euler_r_output_scipy))

if __name__ == '__main__':
    euler_d_input = np.array([20, 80, 15])

    for seq in ROTATION_SEQUENCES:
        test_euler_2_x(euler_d_input, seq)
        break # remove it to test all rotation sequences
