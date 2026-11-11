using TokToxStability
using Test
import GeneralizedPerturbedEquilibrium as GPEC
import TOML

const D3D_GEQDSK = joinpath(pkgdir(TokToxStability), "cases", "D3D", "g147131.02300_DIIID_KEFIT.eqdsk")
const D3D_KIN    = joinpath(pkgdir(TokToxStability), "cases", "D3D", "g147131.02300_DIIID_KEFIT.kin")

# Parse a generated deck with GPEC's own deck reader -- the same path GPEC.main takes -- so any
# key GPEC no longer accepts (renamed/removed between releases) fails here, not mid-scan.
function parse_with_gpec(toml_str)
    dir = mktempdir()
    cp(D3D_GEQDSK, joinpath(dir, basename(D3D_GEQDSK)))
    write(joinpath(dir, "gpec.toml"), toml_str)
    inputs = TOML.parsefile(joinpath(dir, "gpec.toml"))
    return GPEC.parse_deck(inputs, dir)
end

@testset "TokToxStability.jl" begin
    @testset "gpec.toml is accepted by GPEC" begin
        for wall in (false, true, Dict("shape" => "dee", "a" => 0.2, "aw" => 0.05, "bw" => 1.6))
            s = TokToxStability.load_default_settings_kink(1; wall, a=(wall === true ? 0.3 : nothing))
            deck = TokToxStability.make_gpec_toml(basename(D3D_GEQDSK), s.toroidal_mode_number,
                                                  TokToxStability._resolve_wall(s.wall, s.a))
            @test parse_with_gpec(deck) !== nothing
        end
    end

    @testset "wall translation" begin
        @test TokToxStability._wall_toml(Dict("shape" => "nowall")) == ""
        conformal = TOML.parse(TokToxStability._wall_toml(TokToxStability._resolve_wall(true, 0.3)))["Wall"]
        @test conformal == Dict("shape" => "conformal", "standoff_fraction" => 0.3)
        dee = TOML.parse(TokToxStability._wall_toml(Dict("shape" => "dee", "a" => 0.2, "bw" => 1.6)))["Wall"]
        @test dee == Dict("shape" => "dee", "standoff_fraction" => 0.2, "elongation" => 1.6)
        @test_throws ErrorException TokToxStability._resolve_wall(true, nothing)
    end

    @testset "kinetic runs are rejected until wired up" begin
        s = TokToxStability.load_default_settings_kink(1; kinetic_factor=1.0)
        @test_throws ErrorException TokToxStability.run_GPEC(s, D3D_GEQDSK, mktempdir())
    end

    @testset "ideal GPEC run on DIII-D 147131" begin
        dir = mktempdir()
        cp(D3D_GEQDSK, joinpath(dir, basename(D3D_GEQDSK)))
        s = TokToxStability.load_default_settings_tearing(1)
        raw = TokToxStability.run_GPEC(s, basename(D3D_GEQDSK), dir)
        r = TokToxStability.read_output(raw)
        @test r.dW_total ≈ r.dW_plasma + r.dW_vacuum rtol = 1e-6
        # Converged reference values for this g-file (radial-grid study, GPEC v2.0.1); guards against
        # under-resolved grid settings, which shift dW by O(1) without any error.
        @test r.dW_total ≈ 0.924 rtol = 0.01
        @test r.dW_total > 0                       # this reference equilibrium is ideal-stable at n=1
        @test [only(ss.m) for ss in r.singular_surfaces] == [2, 3, 4, 5]
        @test size(r.delta_prime_matrix) == (4, 4)
        dp21 = TokToxStability.delta_prime_for_mode(r, 2, 1)
        @test dp21 ≈ 8.1 rtol = 0.03
        @test ismissing(TokToxStability.delta_prime_for_mode(r, 3, 2))   # n=2 surface absent from an n=1 run

        # Glasser / inner-layer terms through GPEC's SLAYER API
        profiles = TokToxStability.read_kinetic_profiles_from_file(D3D_KIN)
        @test profiles.omega !== nothing
        terms = TokToxStability.compute_glasser_terms(raw, profiles, s)
        @test [(t.m, t.n) for t in terms] == [(2, 1), (3, 1), (4, 1), (5, 1)]
        row = (; r..., glasser_terms=terms)
        dc21 = TokToxStability.delta_crit_for_mode(row, 2, 1)
        @test isfinite(dc21) && dc21 > 0                                  # Glasser effect is stabilizing
        @test TokToxStability.delta_eff_for_mode(row, 2, 1) ≈ dp21 - dc21
    end
end
