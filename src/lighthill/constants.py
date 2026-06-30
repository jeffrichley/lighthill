"""Physical constants and the canonical 6-vector layout for lighthill."""

RHO_SEAWATER: float = 1025.0  # kg/m^3, seawater
GRAVITY: float = 9.81  # m/s^2

# A wrench/twist is [linear(3), angular(3)] in body frame.
LIN = slice(0, 3)
ANG = slice(3, 6)
