# AGENTS Guide Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a concise `AGENTS.md` contributor guide tailored to the C++ development workflow in this repository.

**Architecture:** The work is a docs-only change at the repository root. The guide summarizes the current directory layout, CMake build flow, GTest usage, and contribution norms already visible in the repository so contributors have one fast onboarding reference.

**Tech Stack:** Markdown, CMake, C++20, GoogleTest, MPI, OpenMP

---

### Task 1: Gather repository facts

**Files:**
- Read: `README.md`
- Read: `CMakeLists.txt`
- Read: `.github/workflows/ci.yml`
- Read: `test/*.cpp`

**Step 1: Confirm build and test commands**

Run: `sed -n '1,260p' README.md && sed -n '1,260p' CMakeLists.txt`
Expected: CMake configure/build commands and the `HPDMK_BUILD_TESTS`, `BUILD_BENCHMARK`, and `BUILD_ACCURACY` options are visible.

**Step 2: Confirm testing and naming patterns**

Run: `ls test && sed -n '1,80p' test/hpdmk_test.cpp`
Expected: Test files follow the `*_test.cpp` pattern and use GoogleTest macros.

### Task 2: Draft the contributor guide

**Files:**
- Create: `AGENTS.md`

**Step 1: Write the guide**

Add a 200-400 word Markdown document titled `Repository Guidelines` with sections for structure, commands, style, testing, and contribution workflow.

**Step 2: Keep the scope C++-first**

Document `src/`, `include/`, `test/`, `example/`, `benchmark/`, `accuracy/`, and `extern/` directly. Mention `julia/` only as an additional surface when bindings are touched.

### Task 3: Verify the document

**Files:**
- Test: `AGENTS.md`

**Step 1: Check word count**

Run: `wc -w AGENTS.md`
Expected: Between 200 and 400 words.

**Step 2: Check headings and content**

Run: `sed -n '1,240p' AGENTS.md`
Expected: The title and section headings are present and the commands match the repository.
