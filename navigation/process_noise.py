"""
Process noise models for covariance propagation.

Models unmodeled acceleration uncertainty that causes the true trajectory
to diverge from the propagated trajectory over time.
"""

from __future__ import annotations

import numpy as np


def constant_acceleration_process_noise(q: float, dt: float) -> np.ndarray:
    """Discrete-time process noise matrix for constant acceleration noise.

    Standard 6x6 process noise matrix assuming white noise acceleration
    with power spectral density q.

    Q = q * [[dt^3/3 * I_3, dt^2/2 * I_3],
             [dt^2/2 * I_3, dt   * I_3  ]]

    Args:
        q: Acceleration noise spectral density [km^2/s^5].
            Typical GEO values: 1e-15 (conservative) to 1e-12 (pessimistic).
        dt: Time interval [seconds].

    Returns:
        Q: 6x6 process noise covariance matrix [km^2, km^2/s, km^2/s^2].
    """
    dt2 = dt * dt
    dt3 = dt2 * dt

    Q = np.zeros((6, 6))
    I3 = np.eye(3)

    Q[0:3, 0:3] = (dt3 / 3.0) * I3
    Q[0:3, 3:6] = (dt2 / 2.0) * I3
    Q[3:6, 0:3] = (dt2 / 2.0) * I3
    Q[3:6, 3:6] = dt * I3

    return q * Q
