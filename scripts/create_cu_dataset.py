from ase import units
from ase.build import bulk
from ase.calculators.morse import MorsePotential
from ase.io import write
from ase.md.langevin import Langevin

atoms = bulk("Cu", "fcc", a=3.615, cubic=False).repeat((3, 3, 3))

# Set up the calculator with Morse potential
calc = MorsePotential(epsilon=0.3429, r0=2.866, rho=1.3588, r_cut=6.0)
atoms.calc = calc

# Set initial temperature
T = 300  # Kelvin

# Set random initial velocities according to temperature
MaxwellBoltzmannDistribution = __import__(
    "ase.md.velocitydistribution"
).md.velocitydistribution.MaxwellBoltzmannDistribution
MaxwellBoltzmannDistribution(atoms, temperature_K=T)

# Set up MD simulation
timestep = 2.0 * units.fs  # femtoseconds
friction = 0.01 / units.fs  # friction coefficient
dyn = Langevin(atoms, timestep, temperature_K=T, friction=friction)

print("Starting MD simulation...")
for step in range(50):
    dyn.run(50)

    write("../data/Cu_dataset.xyz", atoms, append=True, format="extxyz")
    print(f"Frame {step}: E = {atoms.get_potential_energy():.3f} eV")

print("Simulation completed. Results saved in Cu_dataset.xyz")
