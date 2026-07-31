# CubeSat Attitude Dynamics

Numerical simulation of the rotational dynamics and attitude of a rigid
CubeSat.

The project is intended to provide a progressive and physically verifiable
implementation of spacecraft attitude dynamics, starting from the torque-free
motion of a rigid body and later extending to environmental torques, actuators,
and attitude control.

## Project status

This project is currently in its initial development phase.

The first objective is to implement and validate:

- quaternion operations;
- Euler's rigid-body equations;
- numerical integration of the rotational state;
- conservation of rotational kinetic energy;
- conservation of angular momentum;
- preservation of the quaternion unit norm.

Orbital dynamics, environmental perturbations, sensors, actuators, and control
laws are outside the initial scope.

## Initial physical model

The CubeSat is initially modeled as a rigid body rotating around its center of
mass.

Its rotational dynamics are governed by Euler's equation:

$$
J\dot{\boldsymbol{\omega}}
+
\boldsymbol{\omega}
\times
J\boldsymbol{\omega}
=
\boldsymbol{\tau},
$$

where:

- $J$ is the inertia matrix expressed in the body frame;
- $\boldsymbol{\omega}$ is the angular velocity expressed in the body frame;
- $\boldsymbol{\tau}$ is the external torque expressed in the body frame.

The angular acceleration is therefore:

$$
\dot{\boldsymbol{\omega}}
=
J^{-1}
\left(
\boldsymbol{\tau}
-
\boldsymbol{\omega}
\times
J\boldsymbol{\omega}
\right).
$$

For the initial torque-free model:

$$
\boldsymbol{\tau} = \boldsymbol{0}.
$$

The attitude kinematics are represented using a unit quaternion:

$$
\dot q
=
\frac{1}{2}
q
\otimes
\begin{bmatrix}
0 \\
\boldsymbol{\omega}
\end{bmatrix},
$$

where $\otimes$ denotes the Hamilton quaternion product.

## Conventions

The following conventions are used throughout the project:

- quaternion format: `[q0, q1, q2, q3]`;
- scalar component first;
- Hamilton quaternion product;
- the quaternion represents the body-to-inertial rotation;
- angular velocity is expressed in the body frame;
- torques are expressed in the body frame;
- the inertia matrix is expressed in the body frame;
- right-handed Cartesian coordinate systems are used;
- SI units are used;
- angular velocities are expressed in radians per second;
- moments of inertia are expressed in kilogram square metres.

These conventions must remain consistent across the equations, implementation,
tests, and visualizations.

## Reference frames

Two main reference frames are considered.

### Inertial frame

The inertial frame is denoted by $I$.

It is treated as fixed for the initial rotational dynamics model.

### Body frame

The body-fixed frame is denoted by $B$.

It is attached to the CubeSat and rotates with it. Whenever possible, its axes
are chosen along the principal axes of inertia of the satellite.

The angular velocity vector is written as:

$$
\boldsymbol{\omega}
=
\begin{bmatrix}
\omega_x \\
\omega_y \\
\omega_z
\end{bmatrix}_B.
$$

## State vector

The complete rotational state contains the attitude quaternion and the angular
velocity:

```text
state = [q0, q1, q2, q3, omega_x, omega_y, omega_z]
```

Mathematically:

$$
\mathbf{x}
=
\begin{bmatrix}
q_0 &
q_1 &
q_2 &
q_3 &
\omega_x &
\omega_y &
\omega_z
\end{bmatrix}^{T}.
$$

The corresponding state derivative is:

$$
\dot{\mathbf{x}}
=
\begin{bmatrix}
\dot q_0 &
\dot q_1 &
\dot q_2 &
\dot q_3 &
\dot \omega_x &
\dot \omega_y &
\dot \omega_z
\end{bmatrix}^{T}.
$$

## Inertia matrix

For a rigid body, the inertia matrix is:

$$
J =
\begin{bmatrix}
J_{xx} & J_{xy} & J_{xz} \\
J_{yx} & J_{yy} & J_{yz} \\
J_{zx} & J_{zy} & J_{zz}
\end{bmatrix}.
$$

When the body-frame axes coincide with the principal axes of inertia, the
matrix is diagonal:

$$
J =
\begin{bmatrix}
J_x & 0 & 0 \\
0 & J_y & 0 \\
0 & 0 & J_z
\end{bmatrix}.
$$

The initial implementation will primarily use diagonal inertia matrices, while
keeping the equations compatible with a general symmetric inertia matrix.

## Quaternion representation

A quaternion is written as:

$$
q =
\begin{bmatrix}
q_0 \\
q_1 \\
q_2 \\
q_3
\end{bmatrix}
=
\begin{bmatrix}
q_0 \\
\mathbf{q}_v
\end{bmatrix},
$$

where:

- $q_0$ is the scalar component;
- $\mathbf{q}_v = [q_1, q_2, q_3]^T$ is the vector component.

A valid attitude quaternion must satisfy:

$$
\lVert q \rVert = 1.
$$

Because numerical integration may introduce a small norm drift, quaternion
normalization may be applied during or after integration. The magnitude of the
drift must also be monitored as a numerical accuracy indicator.

## Conservation laws

For torque-free rigid-body motion, the implementation will be validated using
physical invariants.

### Rotational kinetic energy

The rotational kinetic energy is:

$$
E_{\mathrm{rot}}
=
\frac{1}{2}
\boldsymbol{\omega}^{T}
J
\boldsymbol{\omega}.
$$

In the absence of external torque, this quantity should remain constant.

### Angular momentum

The angular momentum expressed in the body frame is:

$$
\mathbf{H}_B
=
J\boldsymbol{\omega}.
$$

Its components in the rotating body frame may vary with time. However, the
angular momentum vector expressed in the inertial frame must remain constant
for torque-free motion.

Its norm must therefore remain constant:

$$
\lVert \mathbf{H} \rVert
=
\text{constant}.
$$

### Quaternion norm

The quaternion norm must remain equal to one:

$$
\lVert q \rVert = 1.
$$

Small deviations are expected because of numerical integration error, but they
must remain controlled.

## Planned project structure

```text
cubesat-attitude-dynamics/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   └── cubesat_attitude/
│       ├── __init__.py
│       ├── quaternion.py
│       ├── rigid_body.py
│       ├── simulation.py
│       └── diagnostics.py
├── examples/
│   ├── principal_axis_rotation.py
│   └── torque_free_rotation.py
└── tests/
    ├── test_quaternion.py
    ├── test_rigid_body.py
    └── test_conservation.py
```

### Module responsibilities

`quaternion.py`

- quaternion normalization;
- Hamilton product;
- quaternion conjugation;
- conversion to rotation matrices;
- quaternion kinematics.

`rigid_body.py`

- inertia-matrix validation;
- angular momentum computation;
- rotational kinetic energy computation;
- Euler rigid-body equations.

`simulation.py`

- complete state derivative;
- numerical integration;
- simulation configuration;
- result handling.

`diagnostics.py`

- quaternion-norm error;
- energy-conservation error;
- angular-momentum conservation;
- numerical accuracy metrics.

## Installation

### Requirements

- Python 3.12 or later;
- Git;
- a Python virtual environment.

### Clone the repository

```bash
git clone https://github.com/kouoibertrand/cubesat-attitude-dynamics.git
cd cubesat-attitude-dynamics
```

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable installation allows modifications made in `src/` to be used
immediately without reinstalling the package.

## Development commands

Run the test suite:

```bash
pytest
```

Run the static code checks:

```bash
ruff check .
```

Run an example simulation:

```bash
python examples/torque_free_rotation.py
```

The example simulations will be added progressively during development.

## Initial validation cases

The first implementation will be validated using simple cases with known
physical behaviour.

### Case 1: zero angular velocity

Initial condition:

$$
\boldsymbol{\omega}_0
=
\begin{bmatrix}
0 \\
0 \\
0
\end{bmatrix}.
$$

Expected behaviour:

- constant attitude;
- zero angular acceleration;
- zero rotational kinetic energy.

### Case 2: rotation around a principal axis

Initial condition:

$$
\boldsymbol{\omega}_0
=
\begin{bmatrix}
0 \\
0 \\
\omega_z
\end{bmatrix}.
$$

Expected behaviour:

- constant body-frame angular velocity;
- uniform rotation around the selected principal axis;
- constant rotational kinetic energy;
- constant angular momentum;
- quaternion norm equal to one.

### Case 3: general torque-free rotation

Example initial condition:

```python
omega_initial = np.array([0.2, 0.4, 1.0])
```

Expected behaviour:

- body-frame angular-velocity components vary with time;
- rotational kinetic energy remains constant;
- inertial angular momentum remains constant;
- the attitude evolves continuously;
- quaternion norm remains close to one.

The variation of the angular-velocity components is caused by the nonlinear
gyroscopic term:

$$
\boldsymbol{\omega}
\times
J\boldsymbol{\omega}.
$$

### Case 4: intermediate-axis instability

For a rigid body with three distinct principal moments of inertia:

$$
J_x < J_y < J_z,
$$

rotation around the smallest and largest principal inertia axes is stable,
while rotation around the intermediate axis is unstable.

This case will be used to reproduce the tennis-racket theorem numerically.

## Numerical integration

The initial implementation will use `scipy.integrate.solve_ivp`.

The influence of the following parameters will be investigated:

- integration method;
- relative tolerance;
- absolute tolerance;
- integration time step;
- quaternion renormalization;
- simulation duration.

Numerical results will not be considered valid solely because the solver
terminates successfully. They must also satisfy the expected physical
invariants.

## Roadmap

### Phase 1 — Mathematical foundations

- [ ] Implement quaternion normalization.
- [ ] Implement the Hamilton quaternion product.
- [ ] Implement quaternion conjugation.
- [ ] Implement quaternion kinematics.
- [ ] Implement quaternion-to-rotation-matrix conversion.
- [ ] Add unit tests for quaternion operations.

### Phase 2 — Torque-free rigid-body dynamics

- [ ] Implement the inertia matrix.
- [ ] Implement Euler's rigid-body equations.
- [ ] Implement the complete seven-dimensional state derivative.
- [ ] Integrate the state using SciPy.
- [ ] Simulate principal-axis rotation.
- [ ] Simulate general torque-free rotation.

### Phase 3 — Physical validation

- [ ] Monitor quaternion norm.
- [ ] Monitor rotational kinetic energy.
- [ ] Monitor angular-momentum norm.
- [ ] Verify inertial angular-momentum conservation.
- [ ] Study the influence of solver tolerances.
- [ ] Reproduce intermediate-axis instability.

### Phase 4 — Visualization

- [ ] Plot quaternion components.
- [ ] Plot angular-velocity components.
- [ ] Plot conservation errors.
- [ ] Display the CubeSat orientation in three dimensions.
- [ ] Visualize body axes relative to the inertial frame.

### Phase 5 — Environmental torques

- [ ] Gravity-gradient torque.
- [ ] Aerodynamic torque.
- [ ] Solar-radiation-pressure torque.
- [ ] Residual magnetic dipole torque.

### Phase 6 — Sensors and actuators

- [ ] Gyroscope model.
- [ ] Magnetometer model.
- [ ] Sun-sensor model.
- [ ] Magnetorquer model.
- [ ] Reaction-wheel model.

### Phase 7 — Attitude determination and control

- [ ] Attitude-estimation algorithms.
- [ ] Detumbling control.
- [ ] Quaternion feedback control.
- [ ] Proportional-derivative control.
- [ ] Actuator saturation.
- [ ] Closed-loop simulation.

## Design principles

The project follows several development principles:

1. Mathematical conventions must be explicit.
2. Each mathematical operation must be tested independently.
3. Simple analytical cases must be validated before complex simulations.
4. Numerical accuracy must be measured using physical invariants.
5. Physics, numerical integration, visualization, and configuration must remain
   separated.
6. New perturbations or actuators must not be added before the torque-free model
   is validated.
7. All public functions should include units, reference frames, and array shapes
   in their documentation.

## Non-goals of the initial version

The initial version does not attempt to model:

- orbital translation;
- flexible-body dynamics;
- fuel sloshing;
- structural vibrations;
- thermal deformation;
- high-fidelity aerodynamic interactions;
- complete spacecraft flight software;
- hardware-in-the-loop simulation.

These subjects may be considered only after the rigid-body attitude model has
been validated.

## License

No license has been selected yet.

Until a license is added, the repository remains publicly visible but does not
automatically grant permission to reuse, modify, or redistribute the code.
