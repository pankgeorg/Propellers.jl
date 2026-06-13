using Test
using Propellers
using Propellers: in_disk, cell_count
using StaticArrays
using WaterLily

@testset "Propellers" begin

    @testset "ActuatorDisk geometry" begin
        # 2D disk for simple sanity checks
        d = ActuatorDisk(
            center = SVector(0.0, 0.0),
            axis   = SVector(1.0, 0.0),
            R      = 1.0,
            R_hub  = 0.2,
            w      = 0.5,
            thrust = 1.0,
        )

        # Axis is normalized
        @test sum(abs2, d.axis) ≈ 1.0

        # On the disk plane, inside the annulus
        @test in_disk(d, SVector(0.0, 0.5))
        # On the disk plane, inside hub (excluded)
        @test !in_disk(d, SVector(0.0, 0.1))
        # On the disk plane, outside outer R
        @test !in_disk(d, SVector(0.0, 1.5))
        # Off the disk plane (axially) by more than w/2 — excluded
        @test !in_disk(d, SVector(0.4, 0.5))
        # Within axial thickness on the annulus — included
        @test in_disk(d, SVector(0.2, 0.5))

        # Non-axis-aligned disk
        d2 = ActuatorDisk(
            center = SVector(0.0, 0.0),
            axis   = SVector(1.0, 1.0),  # gets normalized
            R      = 1.0,
            w      = 0.2,
            thrust = 1.0,
        )
        @test d2.axis[1] ≈ 1/√2.0
        @test d2.axis[2] ≈ 1/√2.0
    end

    @testset "Thrust conservation on a Flow" begin
        # Build a small 3D Flow and apply the disk via the udf path.
        # Total axial body force in flow.f should equal the prescribed thrust.
        dims = (64, 32, 32)
        uBC = (1f0, 0f0, 0f0)
        flow = WaterLily.Flow(dims, uBC; T=Float32)

        # Disk centred in the domain, oriented along +x
        cx = (dims[1] + 1) / 2 - 1   # `inside` excludes the ghost layer
        cy = (dims[2] + 1) / 2 - 1
        cz = (dims[3] + 1) / 2 - 1
        disk = ActuatorDisk(
            center = SVector(cx, cy, cz),
            axis   = SVector(1.0, 0.0, 0.0),
            R      = 6.0,
            R_hub  = 1.0,
            w      = 2.0,
            thrust = 5f-2,
        )

        # Zero the force buffer (Flow constructor leaves it zeroed already,
        # but be explicit so this test is robust to constructor changes).
        flow.f .= 0
        disk(flow, 0.0)

        # Sum of axial force over the interior should equal thrust.
        total_x = sum(@view flow.f[2:end-1, 2:end-1, 2:end-1, 1])
        total_y = sum(@view flow.f[2:end-1, 2:end-1, 2:end-1, 2])
        total_z = sum(@view flow.f[2:end-1, 2:end-1, 2:end-1, 3])
        @test isapprox(total_x, disk.thrust; rtol = 1e-5)
        @test isapprox(total_y, 0f0; atol = 1e-6)
        @test isapprox(total_z, 0f0; atol = 1e-6)

        # And the disk should have hit a positive number of cells.
        @test cell_count(disk, size(flow.p)) > 0
    end

    @testset "Thrust conservation in 2D" begin
        # 2D analogue: same total axial force should land in flow.f[..,1].
        dims = (48, 24)
        uBC = (1f0, 0f0)
        flow = WaterLily.Flow(dims, uBC; T=Float32)

        cx = (dims[1] + 1) / 2
        cy = (dims[2] + 1) / 2
        disk = ActuatorDisk(
            center = SVector(cx, cy),
            axis   = SVector(1.0, 0.0),
            R      = 4.0,
            R_hub  = 0.5,
            w      = 2.0,
            thrust = 0.1,
        )

        flow.f .= 0
        disk(flow, 0.0)

        total_x = sum(@view flow.f[2:end-1, 2:end-1, 1])
        total_y = sum(@view flow.f[2:end-1, 2:end-1, 2])
        @test isapprox(total_x, disk.thrust; rtol = 1e-5)
        @test isapprox(total_y, 0f0; atol = 1e-6)
        @test cell_count(disk, size(flow.p)) > 0
    end

    @testset "Sign convention" begin
        # Positive thrust accelerates fluid in +axis direction.
        dims = (16, 16, 16)
        uBC = (1f0, 0f0, 0f0)
        flow = WaterLily.Flow(dims, uBC; T=Float32)
        disk = ActuatorDisk(
            center = SVector(7.0, 7.0, 7.0),
            axis   = SVector(0.0, 0.0, 1.0),  # disk pushes fluid in +z
            R      = 4.0,
            w      = 1.0,
            thrust = 1.0,
        )
        flow.f .= 0
        disk(flow, 0.0)
        @test sum(@view flow.f[:,:,:,3]) > 0
        @test isapprox(sum(@view flow.f[:,:,:,1]), 0; atol=1e-6)
        @test isapprox(sum(@view flow.f[:,:,:,2]), 0; atol=1e-6)
    end

    @testset "GradedDisk conservation + radial shape" begin
        dims = (64, 48, 48)
        uBC = (1f0, 0f0, 0f0)
        flow = WaterLily.Flow(dims, uBC; T=Float32)
        cx = (dims[1] + 1) / 2 - 1
        cy = (dims[2] + 1) / 2 - 1
        cz = (dims[3] + 1) / 2 - 1
        # bell-shaped loading (zero hub & tip, peak mid-span)
        rR = collect(0.2:0.1:1.0)
        shapeT = @. sin(pi * (rR - 0.2) / 0.8)
        shapeQ = 0.8 .* shapeT
        disk = GradedDisk(
            center = SVector(cx, cy, cz),
            axis   = SVector(1.0, 0.0, 0.0),
            R = 16.0, R_hub = 3.2, w = 3.0,
            thrust = 12.5, torque = 7.3,
            rR = rR, w_thrust = shapeT, w_torque = shapeQ,
        )
        flow.f .= 0
        disk(flow, 0.0)
        # axial thrust conserved
        @test isapprox(sum(@view flow.f[2:end-1,2:end-1,2:end-1,1]), disk.thrust; rtol=1e-4)
        # torque about +x conserved: Σ (y·f_z − z·f_y) about disk centre
        Q = 0.0
        for I in WaterLily.inside(flow.p)
            x = WaterLily.loc(0, I, Float32)
            y = x[2] - Float32(cy); z = x[3] - Float32(cz)
            Q += y*flow.f[I,3] - z*flow.f[I,2]
        end
        @test isapprox(Q, disk.torque; rtol=1e-3)
        # radial shape: mean axial force in a mid-span annulus should
        # exceed that in a near-hub annulus (bell loading is mid-peaked)
        function annulus_mean_fx(rlo, rhi)
            s = 0.0; n = 0
            for I in WaterLily.inside(flow.p)
                x = WaterLily.loc(0, I, Float32)
                ax = x[1] - Float32(cx)
                abs(ax) > disk.w/2 && continue
                r = sqrt((x[2]-cy)^2 + (x[3]-cz)^2)
                if rlo ≤ r ≤ rhi && flow.f[I,1] != 0
                    s += flow.f[I,1]; n += 1
                end
            end
            n == 0 ? 0.0 : s/n
        end
        @test annulus_mean_fx(8.0, 12.0) > annulus_mean_fx(3.2, 5.0)
    end

end
