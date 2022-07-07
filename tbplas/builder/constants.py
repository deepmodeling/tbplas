"""Constants used through the code."""

NM = 1.0  # Scaling factor from nano-meter to nano-meter
ANG = 0.1  # Scaling factor from Angstrom to nano-meter
BOHR2ANG = 0.529177210671212  # Scaling factor from Bohr to Angstrom
BOHR2NM = BOHR2ANG * ANG  # Scaling factor from Bohr to nano-meter
KB = 8.617333262e-5  # Boltzmann constant in eV/K
HAR2EV = 27.21138602  # Scaling factor from Hartree to eV
H_BAR_EV = 6.582119514e-16  # Reduced Plank constant in eV*s

# Dielectric constant of vacuum in eV/nm/q units
# See lindhard.py for the derivation.
EPSILON0 = 0.6944615417149689
