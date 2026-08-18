# Julia driver for SmoQyDEAC, called by spevae.benchmark.deac.
#
# Deliberately dependency-light on the interface: bins and parameters come in as
# plain delimited text and the spectrum goes out the same way, using only
# DelimitedFiles from the standard library.  The alternative -- exchanging JLD2 /
# HDF5 -- adds a package on the Julia side and a format-compatibility question on
# the Python side, for a payload that is two arrays.
#
# Usage:  julia --project=@deac deac_driver.jl <workdir>
#
# Reads   <workdir>/bins.txt    (N_bins x L_tau, whitespace delimited)
#         <workdir>/params.txt  (key = value, one per line)
# Writes  <workdir>/spectrum.txt  (three columns: omega, A, sigma_A)
#         <workdir>/info.txt      (key = value diagnostics)

using DelimitedFiles
using SmoQyDEAC

function read_params(path)
    p = Dict{String,String}()
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        k, v = split(line, "=", limit = 2)
        p[strip(k)] = strip(v)
    end
    return p
end

function main()
    workdir = ARGS[1]
    p = read_params(joinpath(workdir, "params.txt"))

    bins = readdlm(joinpath(workdir, "bins.txt"), Float64)
    β = parse(Float64, p["beta"])
    nτ = size(bins, 2)

    # SmoQyDEAC wants an evenly spaced tau grid from 0 to beta *inclusive*, which
    # is the grid convention this package uses throughout.
    input_grid = collect(range(0.0, β, length = nτ))

    ωmin = parse(Float64, p["omega_min"])
    ωmax = parse(Float64, p["omega_max"])
    nω = parse(Int, p["n_omega"])
    out_ωs = collect(range(ωmin, ωmax, length = nω))

    num_bins = parse(Int, p["num_bins"])
    runs_per_bin = parse(Int, p["runs_per_bin"])
    generations = parse(Int, p["generations"])
    population = parse(Int, p["population"])
    seed = parse(Int, p["seed"])
    fitness = parse(Float64, p["fitness"])
    find_floor = parse(Bool, p["find_fitness_floor"])

    out_file = joinpath(workdir, "deac_out.jld2")
    chk_file = joinpath(workdir, "deac_chk.jld2")

    t0 = time()
    result = DEAC_Binned(
        bins, β, input_grid, out_ωs, "time_fermionic",
        num_bins, runs_per_bin, out_file, chk_file;
        base_seed = seed,
        number_of_generations = generations,
        population_size = population,
        fitness = fitness,
        find_fitness_floor = find_floor,
        keep_bin_data = false,
        autoresume_from_checkpoint = false,
        verbose = false,
    )
    elapsed = time() - t0

    A = result["A"]
    σ = result["σ"]
    # `data` is (n_omega, n_fitness). SmoQyDEAC sorts the fitness targets
    # descending and, with find_fitness_floor = true, appends the discovered
    # floor -- the smallest chi^2 actually attainable -- at the end (deac.jl,
    # "Remove fitness targets below IFF from array and append IFF value").
    # The last column is therefore the tightest fit, which is the one to report;
    # taking the first would silently return the loosest.
    if ndims(A) > 1
        A = A[:, end]
        σ = σ[:, end]
    end

    writedlm(joinpath(workdir, "spectrum.txt"), hcat(out_ωs, A, σ))
    open(joinpath(workdir, "info.txt"), "w") do io
        println(io, "runtime = ", elapsed)
        println(io, "zeroth_moment = ", result["zeroth_moment"])
        println(io, "zeroth_moment_sigma = ", result["zeroth_moment_σ"])
        println(io, "avg_generations = ", result["avg_generations"])
        println(io, "smoqydeac_version = ", pkgversion(SmoQyDEAC))
    end

    isfile(out_file) && rm(out_file; force = true)
    isfile(chk_file) && rm(chk_file; force = true)
end

main()
