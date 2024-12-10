#!/usr/bin/env python
# coding: utf-8

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem
import ufl
import matplotlib.pyplot as plt
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import basix.ufl

# Create mesh and function space
L = 32.0  # Domain length
<<<<<<< HEAD
N = 512  # Number of elements
=======
N = 256  # Number of elements
>>>>>>> 3611923e1821279b1d34c214fc963765ea77a6ce
domain = mesh.create_interval(MPI.COMM_WORLD, N, [0, L])

# Create function space (using Lagrange elements)
V = fem.functionspace(domain, ("CG", 4))

# Test function
v = ufl.TestFunction(V)

# Functions for solution at current and previous time steps
u_n = fem.Function(V)
u_n1 = fem.Function(V)
u_n_1 = fem.Function(V)

# Initial condition (more pronounced flame front)
def initial_condition(x):
    # Stronger initial perturbation
    return 4.0 * np.exp(-0.5 * ((x[0] - L/2.2)/(L/16))**2) * np.sin(2 * np.pi * x[0] / L)

# Time stepping parameters
dt = fem.Constant(domain, 0.005)
T = 100.0
theta = 0.5
nu = 1.0

# Add stabilization terms
h = fem.Constant(domain, L/N)
stabilization = h*h * ufl.inner(ufl.grad(u_n1), ufl.grad(v))

def linear_operator(u, nu=1.0):
    u_xx = ufl.div(ufl.grad(u))
    u_xxxx = ufl.div(ufl.grad(ufl.div(ufl.grad(u))))
    return - u_xx - (nu * u_xxxx)

def nonlinear_operator(u, m=2.0):
    return - 0.5 * u * ((ufl.grad(u)[0])**m)

# Apply initial condition
x = V.tabulate_dof_coordinates()
u_init = np.zeros(len(x))
u_n_1.x.array[:] = u_init
u_n1.x.array[:] = u_init

for i in range(len(x)):
    u_init[i] = initial_condition(x[i])

u_n.x.array[:] = u_init

# Apply homogeneous Dirichlet BCs at x=0 and x=L
def boundary(x):
<<<<<<< HEAD
    return np.isclose(x[0], L) | np.isclose(x[0], 0)
=======
    return np.isclose(x[0], L)
>>>>>>> 3611923e1821279b1d34c214fc963765ea77a6ce

boundary_facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim-1, boundary)
bdofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim-1, boundary_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), bdofs, V)

# Weak form
F = (u_n1 * v * ufl.dx - u_n * v * ufl.dx +
     dt * theta * linear_operator(u_n1) * v * ufl.dx +
     dt * (1 - theta) * linear_operator(u_n) * v * ufl.dx +
     dt * nonlinear_operator(u_n1) * v * ufl.dx +
     0.001 * stabilization * ufl.dx)

# Create solver with adjusted parameters
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
solution = np.zeros((nt+1, len(x)))
solution[0] = u_init
times = np.linspace(0, T, nt+1)

<<<<<<< HEAD
energy = np.zeros(nt+1)

=======
>>>>>>> 3611923e1821279b1d34c214fc963765ea77a6ce
print("Starting simulation...")
for i in range(nt):
    t += float(dt.value)
    
    if i % 100 == 0:
        print(f"t = {t:.3f}")
    
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
                
<<<<<<< HEAD
        # Compute energy for the new time step
        energy[i+1] = fem.assemble_scalar(fem.form(u_n1 * u_n1 * ufl.dx))

=======
>>>>>>> 3611923e1821279b1d34c214fc963765ea77a6ce
        solution[i+1] = u_n1.x.array[:]
        u_n_1.x.array[:] = u_n.x.array[:]
        u_n.x.array[:] = u_n1.x.array[:]
        
    except Exception as e:
        print(f"Error at t = {t}: {e}")
        break

print("Simulation complete. Creating visualizations...")

# Set up custom colormap for better flame front visualization
from matplotlib.colors import LinearSegmentedColormap
colors = ['darkblue', 'blue', 'lightblue', 'white', 'yellow', 'orange', 'red']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom_fire', colors, N=n_bins)

# Create flame front evolution plot with enhanced visualization
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(15, 10))

# Normalize the solution for better contrast
solution_norm = (solution - np.min(solution)) / (np.max(solution) - np.min(solution))
xx, tt = np.meshgrid(x[:, 0], times)

# Create contour plot with enhanced parameters
levels = np.linspace(0, 1, 50)
cs = ax.contourf(xx, tt, solution_norm, levels=levels, cmap=cmap, extend='both')

# Add colorbar with custom formatting
cbar = fig.colorbar(cs, ax=ax)
cbar.set_label('Normalized Amplitude', color='white', size=12)
cbar.ax.tick_params(colors='white')

# Customize axes
ax.set_xlabel('Spatial Position (x)', fontsize=12, color='white')
ax.set_ylabel('Time (t)', fontsize=12, color='white')
ax.set_title('Kuramoto-Sivashinsky Flame Front Evolution', fontsize=14, pad=20, color='white')
ax.tick_params(colors='white')

# Add grid and adjust layout
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Save high-resolution evolution plot
plt.savefig('ks_flame_evolution_dolfinx.png', dpi=300, bbox_inches='tight', 
            facecolor='black', edgecolor='none')
plt.close()

# Create enhanced snapshots visualization
fig, axes = plt.subplots(4, 1, figsize=(15, 12), facecolor='black')
fig.suptitle('Flame Front Evolution Snapshots', fontsize=16, color='white', y=0.95)

# Select times for snapshots
snapshot_times = [0, int(nt/3), int(2*nt/3), -1]
times_display = [times[i] for i in snapshot_times]

# Plot each snapshot with enhanced styling
for idx, (t_idx, t_val) in enumerate(zip(snapshot_times, times_display)):
    ax = axes[idx]
    ax.set_facecolor('black')
    
    # Plot the flame front
    line = ax.plot(x[:, 0], solution[t_idx], color='orange', lw=2.5)
    ax.fill_between(x[:, 0], solution[t_idx], alpha=0.3, color='red')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    
    # Customize appearance
    ax.grid(True, linestyle='--', alpha=0.2, color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.set_title(f't = {t_val:.1f}', color='white', pad=10)
    
    # Set axis colors
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

# Add x-label to bottom plot only
axes[-1].set_xlabel('Spatial Position (x)', color='white')

# Adjust layout and save
plt.tight_layout()
plt.savefig('ks_flame_snapshots_dolfinx.png', dpi=300, bbox_inches='tight',
            facecolor='black', edgecolor='none')
plt.close()

# Save solution data
np.savez('ks_solution_dolfinx.npz', x=x[:, 0], t=times, solution=solution)

print("\nSimulation Statistics:")
print(f"Maximum value: {np.max(solution):.4f}")
print(f"Minimum value: {np.min(solution):.4f}")
<<<<<<< HEAD
print(f"Final time: {t:.3f}")

# Plot the energy evolution over time
plt.figure(figsize=(10, 6))
plt.plot(times, energy, label="Energy", color="blue")
plt.xlabel("Time (t)")
plt.ylabel("Energy (E)")
plt.title("Energy Evolution in Kuramoto-Sivashinsky Equation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ks_energy_evolution.png", dpi=300)
plt.show()
=======
print(f"Final time: {t:.3f}")
>>>>>>> 3611923e1821279b1d34c214fc963765ea77a6ce
