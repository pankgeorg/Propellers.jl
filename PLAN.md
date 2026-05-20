# PLAN 4 — Propellers.jl

**Repo:** `pankgeorg/Propellers.jl` (private).
**Depends on:** WaterLily upstream-hooks PR (PLAN 1), specifically the
composable-body-force protocol — but the actuator disk also works against
the existing `udf` hook if Hook 3 is shelved, so this package can ship
*first*, before the upstream PR lands.

## Scope

Body-force propeller models that add momentum to a fluid region without
resolving blade geometry:

1. **Uniform actuator disk** — simplest. Constant thrust over a disk
   annulus. Calibrated from a target thrust coefficient `K_T` and advance
   ratio `J`.
2. **Goldstein actuator disk** — radially-varying thrust matching ideal
   actuator-disk theory (Glauert / Goldstein corrections for finite blade
   count).
3. **Actuator line (Sørensen & Shen 2002)** — N rotating 1D line forces
   per blade; each line element applies sectional lift and drag from an
   airfoil table. Far more realistic wake.

## Non-goals

- **No blade-resolved propellers.** That's a different problem (rotating
  immersed body on a fine grid; needs AMR to be tractable). See
  MASTER_PLAN risk register.
- **No coupling to cavitation, structural deflection, or noise.**
- **No** propeller design (inverse problem). This is forward-only.

## API

```julia
using WaterLily, Propellers

disk = Propellers.ActuatorDisk(
    center = SVector(20.0, 0.0, -0.5),
    axis   = SVector(1.0, 0.0, 0.0),
    R      = 0.5,
    R_hub  = 0.1,
    thrust = Propellers.uniform(K_T = 0.2, J = 0.7, n = 10.0, D = 1.0, ρ = 1000.0),
)

line = Propellers.ActuatorLine(
    center = SVector(20.0, 0.0, -0.5),
    axis   = SVector(1.0, 0.0, 0.0),
    R      = 0.5,
    R_hub  = 0.1,
    nBlades = 5,
    Ω      = 2π * 10.0,
    sections = read_propeller_table("KP505.csv"),
)

Simulation(dims, uBC, L; ν, body, forces = (disk,))
# or with upstream Hook 3 shelved:
Simulation(dims, uBC, L; ν, body, udf = Propellers.as_udf(disk))
```

The body force is computed once per `mom_step!` (or once per
`pimple.correct()` if we ever add a PIMPLE loop), distributed to the
nearest grid cells via a Gaussian kernel (actuator line) or a top-hat
(actuator disk).

## Algorithms (primary references)

| Model            | Reference                                                                    |
|------------------|------------------------------------------------------------------------------|
| Uniform AD       | Carrica et al., *J. Ship Res.* 54 (2010) — body-force propeller model in CFD |
| Goldstein AD     | Hough & Ordway, *Princeton Aero Rept* 542 (1965); Glauert (1947)             |
| Actuator line    | Sørensen & Shen, *J. Fluids Eng.* 124 (2002)                                 |
| Gaussian smear   | Martínez-Tossas et al., *Wind Energy* 18 (2015) — ε ≈ 2 Δx                   |

OpenFOAM has `fvModels::propellerDisk` and (via third-party) actuator-line
implementations. **Read for cross-check only; implement from the papers.**

## Validation

### Layer 1 — analytic (momentum theory, fast, per-PR CI)

- **Free-stream actuator disk.** Place a uniform disk in a uniform stream
  with no body. Measure the velocity jump `ΔU` across the disk and the
  downstream slipstream velocity `U_∞ + 2a U_∞`. Compare to 1D actuator
  disk momentum theory:
  - axial induction `a = (1/2)(1 - √(1 - C_T))`
  - thrust coefficient `C_T = 2 ρ A U_∞² a (1 - a)`
  Pass: induced velocity within ±2% of theory; thrust within ±2%.
- **Actuator line in still water, single blade.** Static thrust = ½ ρ A
  C_T (Ω R)² to within ±5% averaged over one revolution.
- **Conservation of momentum.** Net force on the fluid (integrated `∫ρu·dA`
  across box) equals the prescribed thrust within ±0.5%.

### Layer 2 — OpenFOAM tutorial reproduction (nightly CI)

Target: `OpenFOAM/tutorials/incompressibleVoF/DTCHullProp` —
the *exact* problem this package was built for. Tutorial uses a uniform
actuator disk behind the DTC hull (the README explicitly says so).

But that tutorial requires VoF + the hull SDF, so we'd be testing three
things at once. Pre-Phase-3, validate Propellers.jl in isolation against:

Target: `OpenFOAM/tutorials/incompressibleFluid/propeller` (if present in
this OpenFOAM-dev — check; if not, use an external reference case like
the NRL OpenFOAM-Marine `RANS_propellerActuatorDisk` benchmark).

Pass: thrust, torque, and axial induction at the disk plane within ±10%
of OpenFOAM at matching `J`.

### Layer 3 — combined hull + disk vs DTC self-propulsion (release-blocking)

Run DTCHullProp end-to-end inside ShipFlow.jl. Compare:
- self-propulsion point (RPS at which thrust = resistance)
- wake fraction `w`
- thrust deduction `t`

Targets are within ±15% of OpenFOAM and within ±20% of the published
DTC experimental data (el Moctar, Shigunov, Zorn 2012).

## Performance budget

| Component             | Cost vs baseline WaterLily |
|-----------------------|----------------------------|
| Actuator disk         | < 1% per step              |
| Actuator line (N=5)   | < 5% per step              |

These are *cheap*. The body force is touched only inside a small region
of cells; the cost is essentially negligible compared to the rest of the
solver.

## Harness

- Layer 1 tests in `test/runtests.jl`. Includes a small momentum-theory
  rig that runs in < 30 s on CPU.
- Layer 2 + 3 in `test/openfoam/`, nightly via ShipFlow.jl.

## Risks & open questions

- **Force smearing kernel width.** Too narrow ⇒ spurious oscillations.
  Too wide ⇒ blurred wake. Literature recommends ε = 2 Δx for actuator
  line; verify on the free-stream test.
- **Rotation in a fixed reference frame.** Actuator line needs to rotate
  with the blade angular velocity. For a moving ship, that's *two*
  velocities (ship + Ω). Handle via the body's velocity field, not the
  fluid grid (which stays static).
- **Tip-loss correction.** Prandtl tip-loss factor matters at low blade
  count. Glauert/Hough correction handles this; include from the start
  for the Goldstein disk.
- **`udf` vs `BodyForce` decision.** If Hook 3 is shelved upstream, this
  package ships with an `as_udf(::AbstractPropeller)` adapter that
  returns a closure. That keeps the public API unchanged.

## Milestones

| # | Goal                                                | Done when                              |
|---|-----------------------------------------------------|----------------------------------------|
| 1 | Uniform actuator disk passes momentum theory        | Layer 1 disk test green                |
| 2 | Goldstein actuator disk + tip-loss                  | Layer 1 with radial variation         |
| 3 | Actuator line in still water (single blade static)  | thrust ±5%                             |
| 4 | Full actuator line rotating, free stream            | wake structure qualitatively correct   |
| 5 | DTCHullProp end-to-end in ShipFlow.jl               | Layer 3 within ±15% of OpenFOAM        |
