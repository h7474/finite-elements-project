#!/usr/bin/env python
# coding: utf-8

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem
import ufl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import basix.ufl
from scipy.interpolate import griddata

# Create 2D mesh and function space
Lx = Ly = 32.0  # Domain size
Nx = Ny = 128   # Number of elements
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([Lx, Ly])],
    [Nx, Ny],
    cell_type=mesh.CellType.triangle
)

# Create function space (using Lagrange elements)
V = fem.functionspace(domain, ("CG", 2))

# Test function
v = ufl.TestFunction(V)

# Functions for solution at current and previous time steps
u_n = fem.Function(V)
u_n1 = fem.Function(V)
u_n_1 = fem.Function(V)

# Initial condition (2D Gaussian perturbation with sinusoidal modulation)
def initial_condition(x):
    return 4.0 * np.exp(-0.5 * (((x[0] - Lx/2)**2 + (x[1] - Ly/2)**2)/(Lx/8)**2)) * \
           np.sin(2 * np.pi * x[0] / Lx) * np.sin(2 * np.pi * x[1] / Ly)

# Time stepping parameters
dt = fem.Constant(domain, 0.01)
T = 100
theta = 0.5
nu = 1.0

# Add stabilization terms
h = fem.Constant(domain, Lx/Nx)
stabilization = h*h * ufl.inner(ufl.grad(u_n1), ufl.grad(v))

def linear_operator(u, nu=1.0):
    laplacian = ufl.div(ufl.grad(u))
    biharmonic = ufl.div(ufl.grad(ufl.div(ufl.grad(u))))
    return -laplacian - (nu * biharmonic)

def nonlinear_operator(u):
    return -0.5 * (u * ufl.grad(u)[0] + u * ufl.grad(u)[1])

# Apply initial condition
x = V.tabulate_dof_coordinates()
u_init = np.zeros(len(x))
u_n_1.x.array[:] = u_init
u_n1.x.array[:] = u_init

for i in range(len(x)):
    u_init[i] = initial_condition(x[i])

u_n.x.array[:] = u_init

# Apply homogeneous Dirichlet BCs on all boundaries
def boundary(x):
    return np.logical_or.reduce([
        np.isclose(x[0], 0),
        np.isclose(x[0], Lx),
        np.isclose(x[1], 0),
        np.isclose(x[1], Ly)
    ])

boundary_facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim-1, boundary)
bdofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim-1, boundary_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), bdofs, V)

# Weak form
F = (u_n1 * v * ufl.dx - u_n * v * ufl.dx +
     dt * theta * linear_operator(u_n1) * v * ufl.dx +
     dt * (1 - theta) * linear_operator(u_n) * v * ufl.dx +
     dt * nonlinear_operator(u_n1) * v * ufl.dx +
     0.001 * stabilization * ufl.dx)

# Create solver
problem = dolfinx.fem.petsc.NonlinearProblem(F, u_n1, bcs=[bc])
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

# Configure solver parameters
solver.convergence_criterion = "incremental"
solver.rtol = 1e-2
solver.atol = 1e-3
solver.max_it = 100

# Configure PETSc solver
ksp = solver.krylov_solver
opts = PETSc.Options()
opts["ksp_type"] = "gmres"
opts["pc_type"] = "ilu"
opts["ksp_gmres_restart"] = 100
opts["ksp_rtol"] = 1e-4
ksp.setFromOptions()

# Time stepping
nt = int(T/float(dt.value))
t = 0
times = np.linspace(0, T, nt+1)

# Arrays to store solution and energy
solution_snapshots = []
solution_times = []
energy = np.zeros(nt+1)
energy[0] = fem.assemble_scalar(fem.form(u_n * u_n * ufl.dx))

print("Starting simulation...")
for i in range(nt):
    t += float(dt.value)
    
    if i % 10 == 0:
        print(f"t = {t:.3f}")
        solution_snapshots.append(u_n.x.array[:].copy())
        solution_times.append(t)
    
    try:
        for attempt in range(3):
            try:
                num_its, converged = solver.solve(u_n1)
                if converged:
                    break
                u_n1.x.array[:] = 0.5 * (u_n1.x.array[:] + u_n.x.array[:])
            except:
                if attempt == 2:
                    raise
                u_n1.x.array[:] = u_n.x.array[:]
        
        # Compute energy
        energy[i+1] = fem.assemble_scalar(fem.form(u_n1 * u_n1 * ufl.dx))
        
        # Update solution
        u_n_1.x.array[:] = u_n.x.array[:]
        u_n.x.array[:] = u_n1.x.array[:]
        
    except Exception as e:
        print(f"Error at t = {t}: {e}")
        break

print("Simulation complete. Creating visualizations...")

# Set up custom colormap for better visualization
colors = ['darkblue', 'blue', 'lightblue', 'white', 'yellow', 'orange', 'red']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom_fire', colors, N=n_bins)

# Create interpolation grid for visualization
xi = np.linspace(0, Lx, 200)
yi = np.linspace(0, Ly, 200)
Xi, Yi = np.meshgrid(xi, yi)

# Prepare coordinates for griddata
points = x  # These are the mesh node coordinates
values = solution_snapshots[-1]  # Solution values at the mesh nodes

# Create evolution plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(15, 10))

# Interpolate the solution onto regular grid
points = x[:, :2]
grid_z = griddata(points, values, (Xi.flatten(), Yi.flatten()), method='cubic')
grid_z = grid_z.reshape(Xi.shape)

# Plot the interpolated solution
plt.pcolormesh(Xi, Yi, grid_z, cmap=cmap, shading='auto')
plt.colorbar(label='Amplitude')
plt.xlabel('Spatial Position (x)')
plt.ylabel('Spatial Position (y)')
plt.title('2D Kuramoto-Sivashinsky Final State')
plt.tight_layout()
plt.savefig('ks_2d_evolution.png', dpi=300, bbox_inches='tight', 
            facecolor='black', edgecolor='none')
plt.close()

# Create snapshots visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 15), facecolor='black')
fig.suptitle('2D Flame Front Evolution Snapshots', fontsize=16, color='white', y=0.95)

# Select times for snapshots
snapshot_indices = [0, len(solution_snapshots)//3, 2*len(solution_snapshots)//3, -1]
times_display = [solution_times[i] for i in snapshot_indices]

# Plot each snapshot
for idx, (t_idx, t_val) in enumerate(zip(snapshot_indices, times_display)):
    ax = axes[idx//2, idx%2]
    ax.set_facecolor('black')
    
    # Interpolate this snapshot
    values = solution_snapshots[t_idx]
    grid_z = griddata(points, values, (Xi.flatten(), Yi.flatten()), method='cubic')
    grid_z = grid_z.reshape(Xi.shape)
    
    # Plot the snapshot
    im = ax.pcolormesh(Xi, Yi, grid_z, cmap=cmap, shading='auto')
    plt.colorbar(im, ax=ax, label='Amplitude')
    
    # Customize appearance
    ax.set_xlabel('x', color='white')
    ax.set_ylabel('y', color='white')
    ax.set_title(f't = {t_val:.1f}', color='white', pad=10)
    
    # Set axis colors
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

plt.tight_layout()
plt.savefig('ks_2d_snapshots.png', dpi=300, bbox_inches='tight',
            facecolor='black', edgecolor='none')
plt.close()

# Plot energy evolution
plt.style.use('default')
plt.figure(figsize=(10, 6))
plt.plot(times, energy, 'b-', label='Energy')
plt.xlabel('Time (t)')
plt.ylabel('Energy (E)')
plt.title('Energy Evolution in 2D Kuramoto-Sivashinsky Equation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('ks_2d_energy.png', dpi=300)
plt.close()

# Save solution data
np.savez('ks_2d_solution.npz', 
         x=x[:, 0],
         y=x[:, 1],
         t=times,
         snapshots=np.array(solution_snapshots),
         energy=energy)

print("\nSimulation Statistics:")
print(f"Maximum value: {np.max(solution_snapshots[-1]):.4f}")
print(f"Minimum value: {np.min(solution_snapshots[-1]):.4f}")
print(f"Final time: {t:.3f}")