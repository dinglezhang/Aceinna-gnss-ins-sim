import math
import numpy as np
from gnss_ins_sim.attitude import attitude
from scipy.spatial.transform import Rotation

PASSED_STR = "\033[1;32mPASSED\033[0m"       # with green color
FAILED_STR = "\033[1;31mFAILED\033[0m"   # with red color

PASSED_COUNT = 0
FAILED_COUNT = 0

ROTATION_SEQUENCES = (\
    # three axis rotation
    'zyx', 'zxy', 'yxz', 'yzx', 'xyz', 'xzy', \
    # two axis rotation
    'zyz', 'zxz', 'yxy', 'yzy', 'xyx', 'xzx'\
)

def get_result(is_passed):
    global PASSED_COUNT
    global FAILED_COUNT

    if is_passed:
        PASSED_COUNT = PASSED_COUNT + 1
        return PASSED_STR
    else:
        FAILED_COUNT = FAILED_COUNT + 1
        return FAILED_STR

# euler2dcm()
# euler2quat()
# dcm2euler() and quat2euler() by all rotation sequneces. (looks like euler to euler from input)
# dcm2quat() and quat2dcm() as well
def euler_2_x(euler_d_input, rot_seq):
    print('~~~~~~~~~~euler2x~~~~~~~~~~')

    euler_r_input = euler_d_input * attitude.D2R
    print('euler input by %s seqence:'% rot_seq)
    print('%s (%s)\n'% (euler_d_input, euler_r_input))

    rot = Rotation.from_euler(rot_seq.upper(), euler_r_input) # in body frame (intrinsic)

    # test euler2dcm()
    dcm_scipy = rot.as_matrix()
    dcm_att = attitude.euler2dcm(euler_r_input, rot_seq)
    result = get_result(np.allclose(dcm_scipy, dcm_att.T)) # scipy is to rotate vector, but attitude is to rotate frame for postmultiplication
    print('***dcm output: %s***'% result)
    print('scipy: \n%s'% dcm_scipy)
    print('attitude:\n%s\n'% dcm_att)

    # test euler2quat()
    quat_scipy = rot.as_quat()
    quat_att = attitude.euler2quat(euler_r_input, rot_seq)
    result = get_result(np.allclose(quat_scipy, quat_att[[1, 2, 3, 0]]))
    print('***quat output: %s***'% result)
    print('scipy: \n%s'% quat_scipy)
    print('attitude:\n%s\n'% quat_att)

    # test dcm2quat() and quat2dcm() as well
    quat_att = attitude.dcm2quat(dcm_att)
    result = get_result(np.allclose(quat_scipy, quat_att[[1, 2, 3, 0]]))
    print('***dcm2quat() %s***'% result)

    dcm_att = attitude.quat2dcm(quat_att)
    result = get_result(np.allclose(dcm_scipy, dcm_att.T))
    print('***quat2dcm() %s***'% result)

    # test dcm2euler() and quat2euler() for all kinds of rotation sequences
    for seq in ROTATION_SEQUENCES:
        euler_r_scipy = rot.as_euler(seq.upper())
        euler_d_scipy = euler_r_scipy * attitude.R2D

        euler_r_att_dcm = attitude.dcm2euler(dcm_att, seq)
        euler_d_att_dcm = euler_r_att_dcm * attitude.R2D

        euler_r_att_quat = attitude.quat2euler(quat_att, seq)
        euler_d_att_quat = euler_r_att_dcm * attitude.R2D

        result = get_result(np.allclose(euler_r_scipy, euler_r_att_dcm) and np.allclose(euler_r_scipy, euler_r_att_quat))
        print('***euler output by %s sequence: %s***'% (seq, result))
        print('scipy:         %s (%s)'% (euler_d_scipy, euler_r_scipy))
        print('attitude dcm:  %s (%s)'% (euler_d_att_dcm, euler_r_att_dcm))
        print('attitude quat: %s (%s)\n'% (euler_d_att_quat, euler_r_att_quat))

def vectors_rotation(vectors_input, euler_d_input, rot_seq, times = 1, is_extrinsic = False, on_frame = False):
    times_str = ''
    if times > 1:
        times_str = '%s times composed '% times
    way_str = 'intrinsicly'
    if is_extrinsic:
        way_str = 'extrinsicly'
    on_str = 'on vector'
    if on_frame:
        on_str = 'on frame'
    print('~~~~~~~~~~vectors rotation %s%s %s~~~~~~~~~~'% (times_str, way_str, on_str))

    print('vector input:\n%s'% vectors_input)
    euler_r_input = euler_d_input * attitude.D2R
    print('euler input by %s seqence: %s (%s)\n'% (rot_seq, euler_d_input, euler_r_input))

    rot_seq_scipy = rot_seq.upper() # intrinsic
    if is_extrinsic:
        rot_seq_scipy = rot_seq
    rot_once = Rotation.from_euler(rot_seq_scipy, euler_r_input)
    dcm_scipy_once = rot_once.as_matrix()
    dcm_att_once = attitude.euler2dcm(euler_r_input, rot_seq)

    rot = rot_once
    dcm_scipy = dcm_scipy_once
    dcm_att = dcm_att_once
    for i in range(1, times):
        rot = rot * rot_once
        dcm_scipy = dcm_scipy.dot(dcm_scipy_once)
        dcm_att = dcm_att.dot(dcm_att_once)

    vectors_rotated_scipy = rot.apply(vectors_input, on_frame)

    if (not on_frame):
        vectors_rotated_scipy_dcm = (dcm_scipy.dot(vectors_input.T)).T  # premultiplication to rotate vectors
    else:
        vectors_rotated_scipy_dcm = vectors_input.dot(dcm_scipy)        # postmultiplication to rotate frame

    if (not on_frame):
        vectors_rotated_att_dcm = vectors_input.dot(dcm_att)            # postmultiplication to rotate vectors since dcm_att == dcm_scipy.T
    else:
        vectors_rotated_att_dcm = (dcm_att.dot(vectors_input.T)).T      # premultiplication to rotate frame

    is_passed = False
    if np.allclose(vectors_rotated_scipy, vectors_rotated_scipy_dcm):
        if is_extrinsic:  #ignore vectors_rotated_att_dcm result for extrinsic
            is_passed = True
        elif np.allclose(vectors_rotated_scipy, vectors_rotated_att_dcm):
            is_passed = True
    result = get_result(is_passed)

    print('***vetors rotated: %s***'% result)
    print('scipy Rotation:\n%s'% vectors_rotated_scipy)
    print('scipy dcm:\n%s'% vectors_rotated_scipy_dcm)
    if not is_extrinsic:
        print('attitude dcm:\n%s'% vectors_rotated_att_dcm)
    print()

    return vectors_rotated_scipy

def test_euler_2_x():
    print('============================test euler2x============================')

    euler_d_input = np.array([20, 80, 15])
    for seq in ROTATION_SEQUENCES:
        euler_2_x(euler_d_input, seq)
        break # remove it to test all rotation sequences

def test_vectors_rotation(is_extrinsic):
    way_str = 'intrinsicly'
    if is_extrinsic:
        way_str = 'extrinsicly'
    print('============================test vectors rotation %s once============================'% way_str)

    vectors_input = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    euler_d_input = np.array([45, 90, 45])
    vectors_rotation(vectors_input, euler_d_input, 'zyx', 1, is_extrinsic, False)
    vectors_rotation(vectors_input, euler_d_input, 'zyx', 1, is_extrinsic, True)

    print('============================test vectors rotation %s 5 times============================'% way_str)

    euler_d_input = np.array([3, 3, 3])
    times = 5

    vectors_rotated_one_by_one = vectors_input
    for i in range(1, times + 1):
        print('@' + str(i))
        vectors_rotated_one_by_one = vectors_rotation(vectors_rotated_one_by_one, euler_d_input, 'zyx', 1, is_extrinsic, False)

    vectors_rotated_composed = vectors_rotation(vectors_input, euler_d_input, 'zyx', times, is_extrinsic, False)

    result = get_result(np.allclose(vectors_rotated_one_by_one, vectors_rotated_composed))
    print('***vectors rotated results are SAME between one by one and composed: %s***'% result)
    print('one by one:\n%s'% vectors_rotated_one_by_one)
    print('composed:\n%s\n'% vectors_rotated_composed)

    euler_d_input = euler_d_input * times
    vectors_rotated_multiple_angles = vectors_rotation(vectors_input, euler_d_input, 'zyx', 1, is_extrinsic, False)

    result = get_result(not np.allclose(vectors_rotated_one_by_one, vectors_rotated_multiple_angles))
    print('***vectors rotated results are DIFFERENT between one by one and by multiple angles: %s***'% result)
    print('one by one:\n%s'% vectors_rotated_one_by_one)
    print('multiple angles:\n%s\n'% vectors_rotated_multiple_angles)

if __name__ == '__main__':
    test_euler_2_x()
    test_vectors_rotation(False)
    test_vectors_rotation(True)

    print('%s: %s\n%s: %s'% (PASSED_STR, PASSED_COUNT, FAILED_STR, FAILED_COUNT))
