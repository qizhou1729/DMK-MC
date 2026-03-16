# Repository Guidelines

## Project Structure & Module Organization
Core C++ sources live in `src/`; public headers live in `include/`. Put new library logic in matching `src/*.cpp` and `include/*.hpp` or `include/*.h` pairs where practical. Tests live in `test/` and cover the main numerical kernels and tree operations. Use `example/` for runnable demos, `benchmark/` for performance binaries, and `accuracy/` for validation programs. Third-party code is vendored under `extern/`; treat it as external unless a task explicitly targets a submodule. `julia/` contains bindings and tests for the wrapper, but most contributor work should start in the C++ tree.

## Build, Test, and Development Commands
Initialize submodules first: `git submodule update --init --recursive`.
Configure a local build with `cmake -S . -B build`.
Build with `cmake --build build -j 8`.
Run the C++ test suite with `ctest --test-dir build --output-on-failure`.
Enable optional targets when needed: `cmake -S . -B build -DBUILD_BENCHMARK=ON -DBUILD_ACCURACY=ON`.
On cluster environments such as `rusty`, load the required toolchain first: `module load gcc fftw openmpi`.

## Coding Style & Naming Conventions
This project builds as C++20. Match the existing style: 4-space indentation, braces on the same line, and small, focused translation units. Follow existing naming patterns: `snake_case` for files and free functions, `PascalCase` for types such as `HPDMKParams`. Keep MPI/OpenMP-related setup explicit rather than hidden in helper macros. No repo-wide formatter is committed, so format new code to match nearby files and keep diffs tight.

## Testing Guidelines
Add or extend GoogleTest coverage in `test/*_test.cpp`. Prefer deterministic inputs and explicit numeric tolerances for floating-point checks. Run `ctest --test-dir build --output-on-failure` before opening a PR. If a change touches the bindings, also run `julia --project=julia julia/test/runtests.jl`.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `fix ...`, `update ...`, and `add ...`; keep that style and keep each commit focused. Pull requests should explain the numerical or API impact, list the commands you ran, and call out dependency, MPI, or OpenMP assumptions. Include benchmark or accuracy output only when changes affect `benchmark/`, `accuracy/`, or generated figures.
