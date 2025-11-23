module PDMK4MC

using MPI, MPIPreferences
using Logging: @debug
using Base: Cdouble, Cfloat, Cint, Clonglong
import Base: length

export HPDMKParams, hpdmk_init, DIRECT, PROXY, Tree,
       create_tree, destroy_tree!, recontstruct_tree, form_outgoing_pw!, form_incoming_pw!,
       eval_energy, eval_energy_window, eval_energy_diff, eval_energy_res,
       eval_shift_energy, update_shift!, tree_depth, tree_depth_f

const libhpdmk = get(ENV, "HPDMK_LIBRARY", joinpath(@__DIR__, "../../build/libhpdmk.so"))

include("cinterface.jl")
include("molly.jl")

end # module
