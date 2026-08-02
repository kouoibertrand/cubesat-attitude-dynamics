# CubeSat Attitude Dynamics

A learning project for implementing and validating the attitude dynamics of a
rigid CubeSat.

## Current status

The project is being rebuilt progressively from first principles.

Quaternion algebra, vector rotations, and quaternion kinematics are currently
implemented and covered by unit tests.

Rigid-body rotational dynamics and numerical time integration have not yet
been implemented.

## Initial objective

The first objective is to understand and implement the torque-free rotational
motion of a rigid body.

The implementation will be developed incrementally:

1. quaternion representation;
2. elementary quaternion operations;
3. rigid-body rotational dynamics;
4. numerical integration;
5. validation using physical invariants.

## Conventions

The following conventions are currently used:

- quaternion format: `[q0, q1, q2, q3]`;
- scalar component first;
- Hamilton quaternion product;
- body-to-inertial attitude quaternion;
- angular velocity expressed in the body frame;
- right-handed coordinate systems;
- SI units.

These conventions must remain explicit and consistent throughout the project.
Any change must be documented and propagated to the implementation and tests.

## Physical model

The rotational dynamics of a rigid body are governed by Euler's equation:

```math
J\dot{\boldsymbol{\omega}}
+
\boldsymbol{\omega}\times
\left(J\boldsymbol{\omega}\right)
=
\boldsymbol{\tau}
```

The first model will consider torque-free motion:

```math
\boldsymbol{\tau}=\boldsymbol{0}
```

## First development milestone

The first milestone is limited to:

- defining what a quaternion is;
- implementing quaternion normalization;
- testing the implementation using manually verified examples.

## Progress

## Progress

### Quaternion algebra

- [x] Define the quaternion storage convention.
- [x] Implement quaternion normalization.
- [x] Implement quaternion conjugation.
- [x] Implement the Hamilton product.
- [x] Implement quaternion inversion.
- [x] Validate identity, associativity, non-commutativity, norm, and conjugation
      properties.

### Attitude representation and kinematics

- [x] Represent rotations with quaternions.
- [x] Rotate three-dimensional vectors.
- [x] Verify invariance under quaternion sign and scaling.
- [x] Verify quaternion rotation composition order.
- [x] Implement the quaternion derivative from body-frame angular velocity.
- [x] Verify quaternion-derivative tangency and norm properties.

### Rigid-body dynamics

- [ ] Define and validate the inertia matrix.
- [ ] Implement Euler's rigid-body equation.
- [ ] Validate principal-axis rotation.
- [ ] Validate torque-free rotational motion.
- [ ] Monitor rotational kinetic energy and angular momentum.

### Numerical simulation

- [ ] Define the complete rotational state.
- [ ] Integrate angular velocity and attitude over time.
- [ ] Monitor quaternion norm drift.
- [ ] Compare numerical results with analytical cases.

## Development principles

- Every equation must be understood before it is implemented.
- Every function must have a clearly defined mathematical responsibility.
- Every function must be tested independently.
- Simple analytical cases must be validated before numerical simulations.
- Physical conventions, units, and reference frames must remain explicit.