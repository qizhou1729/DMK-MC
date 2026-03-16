# AGENTS Guide Design

## Objective
Create a short contributor guide at the repository root that helps new contributors navigate the C++ codebase, build it, run tests, and follow the existing commit style without copying the full README.

## Scope
The document will be C++-first. It will focus on `src/`, `include/`, `test/`, `example/`, `benchmark/`, and `accuracy/`, plus vendored dependencies in `extern/`. The Julia wrapper in `julia/` will be mentioned only as an optional follow-up surface when a change reaches the bindings.

## Content Plan
The guide will use the required title, `Repository Guidelines`, and include concise sections for project structure, build/test commands, coding style, testing expectations, and commit/PR guidance. Commands will match the current repo workflow: submodule init, CMake configure/build, and `ctest`.

## Style
Keep the file between 200 and 400 words, use short Markdown sections, and prefer direct instructions over policy language. Examples will be concrete shell commands and directory paths from this repository.

## Validation
After writing the file, verify that the word count is within range, headings render cleanly, and each statement matches the current `README.md`, `CMakeLists.txt`, CI workflow, and recent Git history.
