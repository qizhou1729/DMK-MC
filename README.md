# Dual-space multilevel kernel-splitting method for Monte Carlo simulation

This is a C++ implementation of the dual-space multilevel kernel-splitting method for Monte Carlo simulation long-range electrostatics systems.

## How to use

1. Clone the repository
2. Run `git submodule update --init --recursive` to clone the submodules
3. Run `cmake . -B build` to create the build directory
4. Run `cmake --build build -j 8` to build the project with 8 threads
5. Run `ctest --test-dir build` to run the tests

Tips: for usage on `rusty`, need to run `module load gcc fftw openmpi`

## Julia interface

A lightweight Julia wrapper is provided in the [`julia/`](julia) directory. After building the
shared library (`libhpdmk`), activate the Julia project and construct trees directly from Julia:

```julia
using Pkg

Pkg.activate("julia")
using PDMK4MC

params = PDMK4MC.HPDMKParams(L = 20.0)
coords = rand(3, 100)
charges = randn(100)
tree = PDMK4MC.create_tree(coords, charges; params=params)
energy = PDMK4MC.eval_energy(tree)
```

The wrapper expects MPI.jl to load an Open MPI runtime so that its communicators remain compatible
with `libhpdmk`. Configure MPI.jl once via [`MPIPreferences.jl`](https://github.com/JuliaParallel/MPIPreferences.jl)
before importing `PDMK4MC`:

```julia
julia> using MPIPreferences

julia> MPIPreferences.use_jll_binary("OpenMPI_jll")
```

By default the wrapper looks for `libhpdmk` using the standard library search path. Set the
environment variable `HPDMK_LIBRARY` to point to the shared library if it lives in a non-standard
location. The bindings ask `libhpdmk` itself to initialise MPI so that Julia and the native library
always share the same MPI runtime; the optional `comm` keyword can be left as `nothing` to use the
library's `MPI_COMM_WORLD`, or set to an existing communicator (for example `MPI.COMM_WORLD`) so long
as it comes from the same Open MPI installation. The
`precision` keyword selects either `Float64` (default) or `Float32` computations.
