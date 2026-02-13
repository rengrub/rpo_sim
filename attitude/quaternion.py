"""
Quaternion operations.

Convention: q = [q1, q2, q3, q4] where q4 is the scalar component.
Represents rotation from frame A to frame B: v_B = q * v_A * q_conj.

All operations maintain unit norm unless otherwise noted.
"""

from __future__ import annotations

import numpy as np


def q_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication such that R(q1*q2) = R(q1) * R(q2).

    This follows the aerospace (JPL/right-multiply) convention where
    q_multiply(q_A2B, q_B2C) gives q_A2C, and DCMs compose as
    R(q1*q2) = R(q1) * R(q2).

    Args:
        q1: First quaternion [q1,q2,q3, q4_scalar], shape (4,).
        q2: Second quaternion, shape (4,).

    Returns:
        Product quaternion, shape (4,).
    """
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2

    return np.array([
        d1*a2 + a1*d2 - b1*c2 + c1*b2,
        d1*b2 + a1*c2 + b1*d2 - c1*a2,
        d1*c2 - a1*b2 + b1*a2 + c1*d2,
        d1*d2 - a1*a2 - b1*b2 - c1*c2
    ])


def q_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (inverse for unit quaternions).

    Args:
        q: Quaternion, shape (4,).

    Returns:
        Conjugate quaternion [-q1, -q2, -q3, q4], shape (4,).
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])


def q_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit magnitude.

    Args:
        q: Quaternion, shape (4,).

    Returns:
        Unit quaternion, shape (4,).
    """
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([0., 0., 0., 1.])
    return q / n


def q_to_dcm(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion to Direction Cosine Matrix.

    Convention: the quaternion represents a frame rotation from A to B.
    The returned DCM satisfies v_B = R @ v_A (passive / alias rotation).
    This is consistent with q representing ECI-to-body: v_body = R @ v_eci.

    Args:
        q: Unit quaternion [q1,q2,q3, q4_scalar], shape (4,).

    Returns:
        R: 3x3 rotation matrix (DCM) such that v_B = R @ v_A.
    """
    q1, q2, q3, q4 = q

    # Standard passive rotation DCM (Shepperd-consistent):
    #   R(q) such that dcm_to_q(R) recovers q (up to global sign).
    #   Off-diagonal signs follow R[i,j] pattern from
    #   Markley & Crassidis, "Fundamentals of Spacecraft Attitude
    #   Determination and Control", Eq. 2.88.
    R = np.array([
        [1 - 2*(q2**2 + q3**2),  2*(q1*q2 + q3*q4),    2*(q1*q3 - q2*q4)],
        [2*(q1*q2 - q3*q4),      1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q1*q4)],
        [2*(q1*q3 + q2*q4),      2*(q2*q3 - q1*q4),    1 - 2*(q1**2 + q2**2)]
    ])

    return R


def dcm_to_q(R: np.ndarray) -> np.ndarray:
    """Convert DCM to unit quaternion via Shepperd's method.

    Numerically robust for all rotation angles.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Unit quaternion, shape (4,).
    """
    tr = np.trace(R)

    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        q4 = 0.25 / s
        q1 = (R[1, 2] - R[2, 1]) * s
        q2 = (R[2, 0] - R[0, 2]) * s
        q3 = (R[0, 1] - R[1, 0]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q4 = (R[1, 2] - R[2, 1]) / s
        q1 = 0.25 * s
        q2 = (R[0, 1] + R[1, 0]) / s
        q3 = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q4 = (R[2, 0] - R[0, 2]) / s
        q1 = (R[0, 1] + R[1, 0]) / s
        q2 = 0.25 * s
        q3 = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q4 = (R[0, 1] - R[1, 0]) / s
        q1 = (R[0, 2] + R[2, 0]) / s
        q2 = (R[1, 2] + R[2, 1]) / s
        q3 = 0.25 * s

    q = np.array([q1, q2, q3, q4])
    return q_normalize(q)


def q_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Transform a vector using the quaternion's DCM.

    For our convention (ECI-to-body quaternion), this computes
    v_body = R(q) @ v_eci, i.e., expresses v in the body frame.

    Args:
        q: Unit quaternion (frame A to frame B), shape (4,).
        v: 3-vector in frame A, shape (3,).

    Returns:
        The same vector expressed in frame B, shape (3,).
    """
    R = q_to_dcm(q)
    return R @ v


def q_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create quaternion from axis-angle representation.

    Args:
        axis: Rotation axis (unit vector), shape (3,).
        angle: Rotation angle [rad].

    Returns:
        Unit quaternion, shape (4,).
    """
    half = angle / 2.0
    s = np.sin(half)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(half)])


def q_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract axis-angle from a unit quaternion.

    Args:
        q: Unit quaternion, shape (4,).

    Returns:
        axis: Rotation axis (unit vector), shape (3,).
        angle: Rotation angle [rad] in [0, 2pi).
    """
    q = q_normalize(q)
    angle = 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0))
    s = np.sin(angle / 2.0)

    if abs(s) < 1e-10:
        return np.array([0., 0., 1.]), 0.0

    axis = q[0:3] / s
    return axis, angle


def q_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical Linear Interpolation between two quaternions.

    Args:
        q1: Start quaternion, shape (4,).
        q2: End quaternion, shape (4,).
        t: Interpolation parameter in [0, 1].

    Returns:
        Interpolated unit quaternion, shape (4,).
    """
    dot = np.dot(q1, q2)

    # Ensure shortest path
    if dot < 0:
        q2 = -q2
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # Linear interpolation for nearly identical quaternions
        result = q1 + t * (q2 - q1)
        return q_normalize(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    s1 = np.sin((1 - t) * theta) / sin_theta
    s2 = np.sin(t * theta) / sin_theta

    return q_normalize(s1 * q1 + s2 * q2)


def q_angle_between(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular distance between two quaternion orientations.

    Args:
        q1, q2: Unit quaternions, shape (4,).

    Returns:
        Angle [rad] between the two orientations.
    """
    q_diff = q_multiply(q_conjugate(q1), q2)
    _, angle = q_to_axis_angle(q_diff)
    return angle
