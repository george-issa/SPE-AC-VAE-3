# Known-answer DQMC data: the half-filled Hubbard dimer.
#
# Two sites, hopping t = 1, on-site U in the particle-hole symmetric form, mu = 0
# (half filling: no sign problem).  The spectrum is known exactly (four poles with
# temperature-dependent weights, `spevae.data.exact_models.hubbard_cluster`), so
# the binned local G(tau) this script writes is real Monte Carlo data whose
# continuation can be checked against the right answer.
#
#   julia --project=scripts/dqmc scripts/dqmc/hubbard_dimer.jl <U> <beta> <dtau> <N_bins> <bin_size> <outdir> [seed]
#
# Follows the package's hubbard_square tutorial (SmoQyDQMC 2.0.11): Hirsch spin
# decoupling, local + reflection updates, symmetric Trotter propagators.

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf
using Random
using Printf

function run_dimer(; U, β, Δτ, N_bins, bin_size, filepath, seed, N_therm = 2_000, N_updates = 2,
                   n_stab = 10, δG_max = 1e-6, symmetric = true)

    N_measurements = N_bins * bin_size
    datafolder_prefix = @sprintf "hubbard_dimer_U%.2f_b%.2f_dt%.3f" U β Δτ
    simulation_info = SimulationInfo(filepath = filepath, datafolder_prefix = datafolder_prefix,
                                     write_bins_concurrent = false, sID = 1)
    initialize_datafolder(simulation_info)
    rng = Xoshiro(seed)

    # The package requires periodic lattices, so the dimer is ONE unit cell holding two
    # orbitals joined by an intra-cell bond (hopping t = 1): a two-site molecule, no images.
    unit_cell = lu.UnitCell(lattice_vecs = [[1.0]], basis_vecs = [[0.0], [0.5]])
    lattice = lu.Lattice(L = [1], periodic = [true])
    model_geometry = ModelGeometry(unit_cell, lattice)
    bond = lu.Bond(orbitals = (1, 2), displacement = [0])
    bond_id = add_bond!(model_geometry, bond)

    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry, t_bonds = [bond], t_mean = [1.0], t_std = [0.0],
        ϵ_mean = [0.0, 0.0], ϵ_std = [0.0, 0.0], μ = 0.0)
    hubbard_model = HubbardModel(ph_sym_form = true, U_orbital = [1, 2], U_mean = [U, U],
                                 U_std = [0.0, 0.0])
    model_summary(simulation_info = simulation_info, β = β, Δτ = Δτ, model_geometry = model_geometry,
                  tight_binding_model = tight_binding_model, interactions = (hubbard_model,))

    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model, model_geometry = model_geometry, rng = rng)
    hubbard_parameters = HubbardParameters(
        model_geometry = model_geometry, hubbard_model = hubbard_model, rng = rng)
    hst_parameters = HubbardSpinHirschHST(β = β, Δτ = Δτ, hubbard_parameters = hubbard_parameters, rng = rng)

    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)
    initialize_measurements!(measurement_container, tight_binding_model)
    initialize_measurements!(measurement_container, hubbard_model)
    initialize_correlation_measurements!(
        measurement_container = measurement_container, model_geometry = model_geometry,
        correlation = "greens", time_displaced = true, pairs = [(1, 1), (2, 2)])

    fpi_up = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ,
                                 forced_complex_potential = false, forced_complex_kinetic = false)
    fpi_dn = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ,
                                 forced_complex_potential = false, forced_complex_kinetic = false)
    initialize!(fpi_up, fpi_dn, hubbard_parameters)
    initialize!(fpi_up, fpi_dn, hst_parameters)
    Bup = initialize_propagators(fpi_up, symmetric = symmetric, checkerboard = false)
    Bdn = initialize_propagators(fpi_dn, symmetric = symmetric, checkerboard = false)
    fgc_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fgc_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)
    fgc_up_alt = dqmcf.FermionGreensCalculator(fgc_up)
    fgc_dn_alt = dqmcf.FermionGreensCalculator(fgc_dn)
    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))
    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fgc_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fgc_dn)
    Gup_ττ = similar(Gup); Gup_τ0 = similar(Gup); Gup_0τ = similar(Gup)
    Gdn_ττ = similar(Gdn); Gdn_τ0 = similar(Gdn); Gdn_0τ = similar(Gdn)
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    function sweep!()
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn, hst_parameters,
            fermion_path_integral_up = fpi_up, fermion_path_integral_dn = fpi_dn,
            fermion_greens_calculator_up = fgc_up, fermion_greens_calculator_dn = fgc_dn,
            fermion_greens_calculator_up_alt = fgc_up_alt, fermion_greens_calculator_dn_alt = fgc_dn_alt,
            Bup = Bup, Bdn = Bdn, rng = rng)
        (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn, hst_parameters,
            fermion_path_integral_up = fpi_up, fermion_path_integral_dn = fpi_dn,
            fermion_greens_calculator_up = fgc_up, fermion_greens_calculator_dn = fgc_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = false)
        return acceptance_rate
    end

    for n in 1:N_therm
        sweep!()
    end
    acc = 0.0
    for measurement in 1:N_measurements
        for update in 1:N_updates
            acc += sweep!()
        end
        (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = make_measurements!(
            measurement_container,
            logdetGup, sgndetGup, Gup, Gup_ττ, Gup_τ0, Gup_0τ,
            logdetGdn, sgndetGdn, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
            fermion_path_integral_up = fpi_up, fermion_path_integral_dn = fpi_dn,
            fermion_greens_calculator_up = fgc_up, fermion_greens_calculator_dn = fgc_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ,
            model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
            coupling_parameters = (hubbard_parameters, hst_parameters))
        write_measurements!(measurement_container = measurement_container,
                            simulation_info = simulation_info, model_geometry = model_geometry,
                            measurement = measurement, bin_size = bin_size, Δτ = Δτ)
    end
    merge_bins(simulation_info)
    metadata = Dict("U" => U, "beta" => β, "dtau" => Δτ, "N_bins" => N_bins, "bin_size" => bin_size,
                    "N_updates" => N_updates, "N_therm" => N_therm, "seed" => seed,
                    "symmetric" => symmetric, "dG" => δG,
                    "local_acceptance_rate" => acc / (N_measurements * N_updates))
    save_simulation_info(simulation_info, metadata)
    process_measurements(datafolder = simulation_info.datafolder, n_bins = N_bins,
                         export_to_csv = false)
    println("datafolder: ", simulation_info.datafolder)
    return simulation_info.datafolder
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_dimer(U = parse(Float64, ARGS[1]), β = parse(Float64, ARGS[2]), Δτ = parse(Float64, ARGS[3]),
              N_bins = parse(Int, ARGS[4]), bin_size = parse(Int, ARGS[5]), filepath = ARGS[6],
              seed = length(ARGS) > 6 ? parse(Int, ARGS[7]) : 1234)
end
