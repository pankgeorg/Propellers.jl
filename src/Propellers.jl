module Propellers

using WaterLily
using StaticArrays

export ActuatorDisk

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

Diagnostic: number of cells inside the disk over a domain of size
`grid_size::NTuple{D,Int}` (matching the size of `flow.p`).
"""
function cell_count(disk::ActuatorDisk{T,D}, grid_size::NTuple{D,Int}) where {T,D}
    N = 0
    for I in CartesianIndices(grid_size)
        all(2 .≤ Tuple(I) .≤ grid_size .- 1) || continue   # `inside`
        x = WaterLily.loc(0, I, T)
        in_disk(disk, x) && (N += 1)
    end
    return N
end

end # module
