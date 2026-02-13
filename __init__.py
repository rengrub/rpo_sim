"""
RPO Simulation Infrastructure
==============================
Blended medium-high fidelity simulation for GEO Rendezvous and Proximity
Operations (RPO) missions with resolved imaging objectives.

Architecture:
    - HCW for trajectory design initial guesses
    - Numerical ECI propagation with toggleable force models
    - Differential correction to close trajectories in high-fidelity
    - Finite burn conversion with re-correction
    - Linear covariance propagation via STM
    - Quaternion attitude kinematics for sensor pointing
    - Imaging performance assessment and observation window computation
"""

__version__ = "0.1.0"
