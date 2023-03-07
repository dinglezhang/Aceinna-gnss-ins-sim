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
    print('============================test euler2x============================')

    euler_r_input = euler_d_input * attitude.D2R
    print('euler input by %s seqence:'% rot_seq)
    print('%s (%s)\n'% (euler_d_input, euler_r_input))

    rot = Rotation.from_euler(rot_seq.upper(), euler_r_input) # in body frame (intrinsic)

    # test euler2dcm()
    dcm_scipy = rot.as_matrix()
    dcm_att = attitude.euler2dcm(euler_r_input, rot_seq)
    result = FAILED_STR
    if np.allclose(dcm_scipy, dcm_att.T): # scipy is to rotate vector, but attitude is to rotate frame for postmultiplication
        result =  PASS_STR
    print('***dcm output: %s***'% result)
    print('scipy: \n%s'% dcm_scipy)
    print('attitude:\n%s\n'% dcm_att)

    # test euler2quat()
    quat_scipy = rot.as_quat()
    quat_att = attitude.euler2quat(euler_r_input, rot_seq)
    result = FAILED_STR
    if np.allclose(quat_scipy, quat_att[[1, 2, 3, 0]]):
        result =  PASS_STR
    print('***quat output: %s***'% result)
    print('scipy: \n%s'% quat_scipy)
    print('attitude:\n%s\n'% quat_att)

    # test dcm2quat() and quat2dcm() as well
    quat_att = attitude.dcm2quat(dcm_att)
    if not np.allclose(quat_scipy, quat_att[[1, 2, 3, 0]]):
        print('***dcm2quat() %s***'% FAILED_STR)
        return
    dcm_att = attitude.quat2dcm(quat_att)
    if not np.allclose(dcm_scipy, dcm_att.T):
        print('***quat2dcm() %s***'% FAILED_STR)
        return

    # test dcm2euler() and quat2euler() for all kinds of rotation sequences
    for seq in ROTATION_SEQUENCES:
        euler_r_scipy = rot.as_euler(seq.upper())
        euler_d_scipy = euler_r_scipy * attitude.R2D

        euler_r_att_dcm = attitude.dcm2euler(dcm_att, seq)
        euler_d_att_dcm = euler_r_att_dcm * attitude.R2D

        euler_r_att_quat = attitude.quat2euler(quat_att, seq)
        euler_d_att_quat = euler_r_att_dcm * attitude.R2D

        result = FAILED_STR
        if np.allclose(euler_r_scipy, euler_r_att_dcm) and \
           np.allclose(euler_r_scipy, euler_r_att_quat):
            result =  PASS_STR

        print('***euler output by %s sequence: %s***'% (seq, result))
        print('scipy:         %s (%s)'% (euler_d_scipy, euler_r_scipy))
        print('attitude dcm:  %s (%s)'% (euler_d_att_dcm, euler_r_att_dcm))
        print('attitude quat: %s (%s)\n'% (euler_d_att_quat, euler_r_att_quat))

def test_vectors_rotation_intrinsic(vectors_input, euler_d_input, rot_seq, on_frame):
    on_str = 'on vector'
    if on_frame:
        on_str = 'on frame'
    print('============================test vectors rotation intrinsicly %s============================'% on_str)

    print('vector input:\n%s'% vectors_input)

    euler_r_input = euler_d_input * attitude.D2R
    print('euler input by %s seqence: %s (%s)\n'% (rot_seq, euler_d_input, euler_r_input))

    rot = Rotation.from_euler(rot_seq.upper(), euler_r_input)
    vectors_rotated_scipy = rot.apply(vectors_input, on_frame)

    dcm_scipy = rot.as_matrix()
    if (not on_frame):
        vectors_rotated_scipy_dcm = (dcm_scipy.dot(vectors_input.T)).T  # premultiplication to rotate vectors
    else:
        vectors_rotated_scipy_dcm = vectors_input.dot(dcm_scipy)        # postmultiplication to rotate frame

    dcm_att = attitude.euler2dcm(euler_r_input, rot_seq)
    if (not on_frame):
        vectors_rotated_att_dcm = vectors_input.dot(dcm_att)            # postmultiplication to rotate vectors since dcm_att == dcm_scipy.T
    else:
        vectors_rotated_att_dcm = (dcm_att.dot(vectors_input.T)).T      # premultiplication to rotate frame

    result = FAILED_STR
    if np.allclose(vectors_rotated_scipy, vectors_rotated_scipy_dcm) and \
       np.allclose(vectors_rotated_scipy, vectors_rotated_att_dcm):
        result =  PASS_STR

    print('***vetors rotated: %s***'% result)
    print('scipy Rotation:\n%s'% vectors_rotated_scipy)
    print('scipy dcm:\n%s'% vectors_rotated_scipy_dcm)
    print('attitude dcm:\n%s\n'% vectors_rotated_att_dcm)

# only scipy has function like this
def test_vectors_rotation_extrinsic(vectors_input, euler_d_input, rot_seq, on_frame):
    on_str = 'on vector'
    if on_frame:
        on_str = 'on frame'
    print('============================test vectors rotation extrinsicly %s============================'% on_str)

    print('vector input:\n%s'% vectors_input)

    euler_r_input = euler_d_input * attitude.D2R
    print('euler input by %s seqence: %s (%s)\n'% (rot_seq, euler_d_input, euler_r_input))

    rot = Rotation.from_euler(rot_seq, euler_r_input)
    vectors_rotated_scipy = rot.apply(vectors_input, on_frame)

    dcm_scipy = rot.as_matrix()
    if (not on_frame):
        vectors_rotated_scipy_dcm = (dcm_scipy.dot(vectors_input.T)).T  # premultiplication to rotate vectors
    else:
        vectors_rotated_scipy_dcm = vectors_input.dot(dcm_scipy)        # postmultiplication to rotate frame

    result = FAILED_STR
    if np.allclose(vectors_rotated_scipy, vectors_rotated_scipy_dcm):
        result =  PASS_STR

    print('***vetors rotated: %s***'% result)
    print('scipy Rotation:\n%s'% vectors_rotated_scipy)
    print('scipy dcm:\n%s\n'% vectors_rotated_scipy_dcm)

if __name__ == '__main__':
    euler_d_input = np.array([20, 80, 15])
    for seq in ROTATION_SEQUENCES:
        test_euler_2_x(euler_d_input, seq)
        break # remove it to test all rotation sequences

    euler_d_input = np.array([45, 90, 45])
    vectors_input = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    test_vectors_rotation_intrinsic(vectors_input, euler_d_input, 'zyx', False)
    test_vectors_rotation_intrinsic(vectors_input, euler_d_input, 'zyx', True)

    test_vectors_rotation_extrinsic(vectors_input, euler_d_input, 'zyx', False)
    test_vectors_rotation_extrinsic(vectors_input, euler_d_input, 'zyx', True)
