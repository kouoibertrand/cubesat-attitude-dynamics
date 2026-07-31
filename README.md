# CubeSat Attitude Dynamics

A learning project for implementing and validating the attitude dynamics of a
rigid CubeSat.

## Current status

The project is being rebuilt progressively from first principles.

No complete attitude simulation has been implemented yet.

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

The following conventions will initially be used:

- quaternion format: `[q0, q1, q2, q3]`;
- scalar component first;
- Hamilton quaternion product;
- body-to-inertial attitude quaternion;
- angular velocity expressed in the body frame;
- right-handed coordinate systems;
- SI units.

These conventions may be refined as the mathematical model is developed.

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

## Development principles

- Every equation must be understood before it is implemented.
- Every function must have a clearly defined mathematical responsibility.
- Every function must be tested independently.
- Simple analytical cases must be validated before numerical simulations.
- Physical conventions, units, and reference frames must remain explicit.