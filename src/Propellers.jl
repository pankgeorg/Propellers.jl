module Propellers

using WaterLily
using StaticArrays

export ActuatorDisk, SwirlingDisk

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

end # module
