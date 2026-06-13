module Propellers

using WaterLily
using StaticArrays

export ActuatorDisk, SwirlingDisk, GradedDisk

"""
    ActuatorDisk(; center, axis, R, R_hub=0, w=1, thrust)

Uniform-thrust actuator disk as a body-force model. Plug into a WaterLily
`Simulation` via the existing `udf` hook:

```julia
disk = ActuatorDisk(
    center = SVector(20f0, 16f0, 16f0),
    axis   = SVector(1f0, 0f0, 0f0),
    R      = 4f0, R_hub = 1f0, w = 2f0,
    thrust = 5f-2,    # total axial thrust in non-dim (ρ=1) units
)

sim_step!(sim; udf = disk)
```

The disk occupies a thin cylindrical annulus around `center`, normal to
`axis` (must be a unit vector — it is normalized on construction), with
inner radius `R_hub`, outer radius `R`, and axial thickness `w` (in
grid cells, default 1). The total axial body force integrated over the
disk volume equals `thrust` by construction; per-cell force is uniform
inside the annulus and zero outside.

Force is added to `flow.f`, matching the convention used by
`WaterLily.accelerate!`. Sign convention: positive `thrust` accelerates
the fluid in the `+axis` direction (i.e. propels the body in `-axis`).
"""
struct ActuatorDisk{T,D,V<:SVector{D,T}}
    center::V
    axis::V
    R::T
    R_hub::T
    w::T
    thrust::T
end

function ActuatorDisk(; center, axis, R::Real, R_hub::Real=zero(R),
                       w::Real=one(R), thrust::Real)
    D = length(center)
    @assert length(axis) == D "center and axis must have the same dimension"
    a = SVector{D}(float.(axis))
    n = sqrt(sum(abs2, a))
    @assert n > 0 "axis must be non-zero"
    a = a ./ n
    c = SVector{D}(float.(center))
    T = promote_type(eltype(c), eltype(a), typeof(float(R)),
                     typeof(float(R_hub)), typeof(float(w)),
                     typeof(float(thrust)))
    ActuatorDisk{T,D,SVector{D,T}}(
        SVector{D,T}(c), SVector{D,T}(a),
        T(R), T(R_hub), T(w), T(thrust),
    )
end

"""
    in_disk(disk, x) -> Bool

Predicate: is location `x` inside the disk's annular region?
"""
@inline function in_disk(disk::ActuatorDisk, x)
    Δ = x .- disk.center
    axial = sum(Δ .* disk.axis)
    abs(axial) > disk.w/2 && return false
    r² = sum(abs2, Δ) - axial^2
    return disk.R_hub^2 ≤ r² ≤ disk.R^2
end

"""
    disk(flow, t)

`udf`-compatible call: add the disk's body force into `flow.f`. Uses a
top-hat distribution over the cells whose centers fall inside the
annulus; per-cell magnitude is `thrust / N_cells`, so the discrete sum
exactly equals the prescribed thrust.
"""
function (disk::ActuatorDisk{T,D})(flow, t; kwargs...) where {T,D}
    Tf = eltype(flow.f)
    # First pass: count cells inside the disk.
    N = 0
    for I in WaterLily.inside(flow.p)
        x = WaterLily.loc(0, I, Tf)
        in_disk(disk, x) && (N += 1)
    end
    N == 0 && return nothing
    f_mag = Tf(disk.thrust / N)
    # Second pass: apply force in axis direction.
    for I in WaterLily.inside(flow.p)
        x = WaterLily.loc(0, I, Tf)
        in_disk(disk, x) || continue
        for i in 1:D
            flow.f[I, i] += f_mag * disk.axis[i]
        end
    end
    return nothing
end

"""
    cell_count(disk, grid_size) -> Int

Diagnostic: number of cells inside the disk over a domain whose **flow
pressure array** is sized `grid_size` (i.e. pass `size(flow.p)`, not the
constructor's `dims` — `flow.p` is `dims .+ 2` because of the WaterLily
ghost layer). The iteration matches the `WaterLily.inside(flow.p)` rule
the runtime force loop uses, so this number is *exactly* the divisor
the runtime applies to `disk.thrust`.
"""
function cell_count(disk::ActuatorDisk{T,D}, grid_size::NTuple{D,Int}) where {T,D}
    N = 0
    for I in CartesianIndices(ntuple(d -> 2:grid_size[d] - 1, D))
        x = WaterLily.loc(0, I, T)
        in_disk(disk, x) && (N += 1)
    end
    return N
end

"""
    SwirlingDisk(; center, axis, R, R_hub=0, w=1, thrust, torque)

Actuator disk with axial thrust **and** torque about `axis` — a single-pass
surrogate for a rotating propeller without resolving the blades. Axial
force is top-hat uniform (matches `ActuatorDisk`). Tangential force is
linear in radius (canonical propeller-blade loading), normalised so the
discrete sum of `r × f_θ` equals `torque`.

Sign convention: positive `torque` rotates the fluid in the right-hand
sense about `+axis`. For a real propeller this would be the sense in
which the propeller itself spins.

Only available in 3D. (2D has no meaningful axial torque.)
"""
struct SwirlingDisk{T,V<:SVector{3,T}}
    center::V
    axis::V
    R::T
    R_hub::T
    w::T
    thrust::T
    torque::T
end

function SwirlingDisk(; center, axis, R::Real, R_hub::Real=zero(R),
                       w::Real=one(R), thrust::Real, torque::Real)
    @assert length(center) == 3 && length(axis) == 3 "SwirlingDisk is 3D only"
    a = SVector{3}(float.(axis))
    n = sqrt(sum(abs2, a))
    @assert n > 0 "axis must be non-zero"
    a = a ./ n
    c = SVector{3}(float.(center))
    T = promote_type(eltype(c), eltype(a), typeof(float(R)),
                     typeof(float(R_hub)), typeof(float(w)),
                     typeof(float(thrust)), typeof(float(torque)))
    SwirlingDisk{T, SVector{3,T}}(
        SVector{3,T}(c), SVector{3,T}(a),
        T(R), T(R_hub), T(w), T(thrust), T(torque),
    )
end

"""
    _disk_geometry(disk, x) -> (inside::Bool, r::T, t_vec::SVector{3,T})

Compute the local disk-relative geometry at world point `x`: whether
`x` lies inside the swept annulus, the radial distance `r` from the
axis, and the tangential unit vector `t_vec = axis × r̂` (zero at the
hub centre). Used by the SwirlingDisk callable to apply axial + swirl
forces.
"""
@inline function _disk_geometry(disk::SwirlingDisk{T}, x) where T
    Δ = x .- disk.center
    axial = sum(Δ .* disk.axis)
    if abs(axial) > disk.w/2
        return (false, zero(T), SVector{3,T}(0,0,0))
    end
    r_vec = Δ .- axial .* disk.axis
    r² = sum(abs2, r_vec)
    if r² < disk.R_hub^2 || r² > disk.R^2
        return (false, zero(T), SVector{3,T}(0,0,0))
    end
    r = sqrt(r²)
    # Tangential unit vector = axis × r̂. If r is too small (hub center),
    # tangential direction is undefined — set to zero.
    if r < eps(T)
        return (true, r, SVector{3,T}(0,0,0))
    end
    r_hat = r_vec ./ r
    t_vec = SVector{3,T}(
        disk.axis[2]*r_hat[3] - disk.axis[3]*r_hat[2],
        disk.axis[3]*r_hat[1] - disk.axis[1]*r_hat[3],
        disk.axis[1]*r_hat[2] - disk.axis[2]*r_hat[1],
    )
    return (true, r, t_vec)
end

"""
    disk(flow, t)  (SwirlingDisk)

`udf`-compatible call: add axial thrust + tangential swirl into
`flow.f`. Axial force is top-hat uniform (per-cell magnitude
`thrust / N_cells`); tangential force scales linearly with radius
(`f_θ = K_τ · r`), with `K_τ` chosen so that `Σ r·f_θ = torque` after
discretisation. The two passes are necessary because `K_τ` depends on
the cell count `N` and the sum of `r²` inside the annulus.
"""
function (disk::SwirlingDisk{T})(flow, t; kwargs...) where T
    Tf = eltype(flow.f)
    # First pass: count cells, sum r² for torque normalisation.
    N = 0
    sum_r² = zero(Tf)
    for I in WaterLily.inside(flow.p)
        x = WaterLily.loc(0, I, Tf)
        inside, r, _ = _disk_geometry(disk, x)
        inside || continue
        N += 1
        sum_r² += r*r
    end
    N == 0 && return nothing
    f_axial = Tf(disk.thrust / N)
    K_τ = sum_r² > 0 ? Tf(disk.torque / sum_r²) : zero(Tf)
    # Second pass: apply axial + tangential force.
    for I in WaterLily.inside(flow.p)
        x = WaterLily.loc(0, I, Tf)
        inside, r, t_vec = _disk_geometry(disk, x)
        inside || continue
        f_tan = K_τ * r
        @inbounds for i in 1:3
            flow.f[I, i] += f_axial * disk.axis[i] + f_tan * t_vec[i]
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# GradedDisk — radially-graded actuator disk
# ----------------------------------------------------------------------------

"""
    GradedDisk(; center, axis, R, R_hub=0, w=1, thrust, torque=0,
                 rR, w_thrust, w_torque)

Actuator disk whose axial thrust and tangential torque follow a
prescribed **radial loading shape** rather than a top-hat. `rR` is a
vector of radial stations (in `r/R`, increasing) and `w_thrust`,
`w_torque` are the (relative) sectional loadings `dT/dr`, `dQ/dr` at
those stations — e.g. the bell-shaped distribution from a propeller
vortex-lattice solve. The shapes are interpolated to each cell's radius,
then discretely normalised so that

  Σ_cells f_axial            = thrust
  Σ_cells r · f_tangential   = torque

to machine precision (same exact-conservation guarantee as
`SwirlingDisk`). The tangential force per cell is `f_θ ∝ (dQ/dr)(r)/r`
so that its moment `r·f_θ ∝ dQ/dr` reproduces the prescribed torque
distribution; the axial force per cell is `f_x ∝ (dT/dr)(r)`.

Sign convention matches `SwirlingDisk`: positive `thrust` accelerates
the fluid along `+axis`; positive `torque` swirls it right-handed about
`+axis`. 3D only.
"""
struct GradedDisk{T,V<:SVector{3,T},Vec<:AbstractVector{T}}
    center::V
    axis::V
    R::T
    R_hub::T
    w::T
    thrust::T
    torque::T
    rR::Vec
    w_thrust::Vec
    w_torque::Vec
end

function GradedDisk(; center, axis, R::Real, R_hub::Real=zero(R),
                     w::Real=one(R), thrust::Real, torque::Real=zero(thrust),
                     rR::AbstractVector, w_thrust::AbstractVector,
                     w_torque::AbstractVector=zero(w_thrust))
    @assert length(center) == 3 && length(axis) == 3 "GradedDisk is 3D only"
    @assert length(rR) == length(w_thrust) == length(w_torque) "rR/w_thrust/w_torque length mismatch"
    a = SVector{3}(float.(axis)); n = sqrt(sum(abs2, a))
    @assert n > 0 "axis must be non-zero"
    a = a ./ n
    c = SVector{3}(float.(center))
    T = promote_type(eltype(c), eltype(a), typeof(float(R)),
                     typeof(float(R_hub)), typeof(float(w)),
                     typeof(float(thrust)), typeof(float(torque)),
                     eltype(float.(rR)), eltype(float.(w_thrust)),
                     eltype(float.(w_torque)))
    GradedDisk{T,SVector{3,T},Vector{T}}(
        SVector{3,T}(c), SVector{3,T}(a),
        T(R), T(R_hub), T(w), T(thrust), T(torque),
        T.(collect(rR)), T.(collect(w_thrust)), T.(collect(w_torque)),
    )
end

# linear interpolation of the loading shape at radius fraction x = r/R
@inline function _grad_interp(rR, ys, x::T) where T
    x ≤ rR[1]   && return ys[1]
    x ≥ rR[end] && return ys[end]
    @inbounds for i in 1:length(rR)-1
        if rR[i] ≤ x ≤ rR[i+1]
            t = (x - rR[i]) / (rR[i+1] - rR[i])
            return ys[i] + t*(ys[i+1] - ys[i])
        end
    end
    return ys[end]
end

@inline function _graded_geometry(disk::GradedDisk{T}, x) where T
    Δ = x .- disk.center
    axial = sum(Δ .* disk.axis)
    if abs(axial) > disk.w/2
        return (false, zero(T), SVector{3,T}(0,0,0))
    end
    r_vec = Δ .- axial .* disk.axis
    r² = sum(abs2, r_vec)
    if r² < disk.R_hub^2 || r² > disk.R^2
        return (false, zero(T), SVector{3,T}(0,0,0))
    end
    r = sqrt(r²)
    r < eps(T) && return (true, r, SVector{3,T}(0,0,0))
    r_hat = r_vec ./ r
    t_vec = SVector{3,T}(
        disk.axis[2]*r_hat[3] - disk.axis[3]*r_hat[2],
        disk.axis[3]*r_hat[1] - disk.axis[1]*r_hat[3],
        disk.axis[1]*r_hat[2] - disk.axis[2]*r_hat[1],
    )
    return (true, r, t_vec)
end

function (disk::GradedDisk{T})(flow, t; kwargs...) where T
    Tf = eltype(flow.f)
    # First pass: accumulate the discrete normalisers.
    #   axial:      Σ shape_T(r)            (so f_x = thrust·shape_T/ΣT)
    #   tangential: Σ r·(shape_Q(r)/r) = Σ shape_Q(r)   (so Σ r·f_θ = torque)
    sumT = zero(Tf); sumQ = zero(Tf)
    @inbounds for I in WaterLily.inside(flow.p)
        x = WaterLily.loc(0, I, Tf)
        inside, r, _ = _graded_geometry(disk, x)
        inside || continue
        xr = Tf(r / disk.R)
        sumT += Tf(_grad_interp(disk.rR, disk.w_thrust, xr))
        sumQ += Tf(_grad_interp(disk.rR, disk.w_torque, xr))
    end
    sumT == 0 && return nothing
    kT = Tf(disk.thrust) / sumT
    kQ = sumQ > 0 ? Tf(disk.torque) / sumQ : zero(Tf)
    # Second pass: apply.
    @inbounds for I in WaterLily.inside(flow.p)
        x = WaterLily.loc(0, I, Tf)
        inside, r, t_vec = _graded_geometry(disk, x)
        inside || continue
        xr = Tf(r / disk.R)
        f_axial = kT * Tf(_grad_interp(disk.rR, disk.w_thrust, xr))
        # f_θ chosen so r·f_θ = kQ·shape_Q(r)  ⇒  Σ r·f_θ = torque
        f_tan = (r > eps(Tf)) ? kQ * Tf(_grad_interp(disk.rR, disk.w_torque, xr)) / r : zero(Tf)
        for i in 1:3
            flow.f[I, i] += f_axial * disk.axis[i] + f_tan * t_vec[i]
        end
    end
    return nothing
end

end # module
