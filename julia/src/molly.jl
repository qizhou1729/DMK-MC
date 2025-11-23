# extend Molly's MC process
using Molly
using Molly.Random

export MetropolisMonteCarloPDMK

struct MetropolisMonteCarloPDMK{T, M, TE}
    temperature::T
    trial_moves::M
    trial_args::Dict
    tree::Tree
    eps_r::TE
end

function MetropolisMonteCarloPDMK(; temperature, trial_moves, trial_args, tree::Tree, eps_r)
    return MetropolisMonteCarloPDMK(temperature, trial_moves, trial_args, tree, eps_r)
end

@inline function Molly.simulate!(sys::System,
                           sim::MetropolisMonteCarloPDMK,
                           n_steps::Integer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           rng=Random.default_rng())
    neighbors = Molly.find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    buffers = Molly.init_buffers!(sys, n_threads)
    E_old = Molly.potential_energy(sys, neighbors, buffers; n_threads=n_threads)
    PDMK4MC.form_outgoing_pw!(sim.tree)
    PDMK4MC.form_incoming_pw!(sim.tree)
    E_pdmk_old = PDMK4MC.eval_energy(sim.tree) * 138.935457644u"kJ/mol" / sim.eps_r

    for step_n in 1:n_steps
        idx, shift_vec = sim.trial_moves(sys; sim.trial_args...)
        # @show idx, shift_vec
        sys.coords[idx] = Molly.wrap_coords(sys.coords[idx] .+ shift_vec, sys.boundary)

        neighbors = Molly.find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
        E_new = Molly.potential_energy(sys, neighbors, buffers, step_n; n_threads=n_threads)

        dE_pdmk = PDMK4MC.eval_shift_energy(sim.tree, idx, shift_vec[1].val, shift_vec[2].val, shift_vec[3].val) * 138.935457644u"kJ/mol" / sim.eps_r

        ΔE = E_new - E_old + dE_pdmk
        δ = ΔE / (sys.k * sim.temperature)
        # println("δ = $(δ), ΔE = $(ΔE), E_new = $(E_new), E_old = $(E_old), dE_pdmk = $(dE_pdmk)")
        if δ < 0 || (rand(rng) < exp(-δ))
            Molly.apply_loggers!(sys, nothing, neighbors, step_n, run_loggers; n_threads=n_threads,
                           current_potential_energy=(E_new + E_pdmk_old + dE_pdmk), success=true,
                           energy_rate=((E_new + E_pdmk_old + dE_pdmk) / (sys.k * sim.temperature)))
            PDMK4MC.update_shift!(sim.tree, idx, shift_vec[1].val, shift_vec[2].val, shift_vec[3].val)
            E_old = E_new
            E_pdmk_old += dE_pdmk

            println("E_lj = $(E_old), E_elec = $(E_pdmk_old)")
        else
            sys.coords[idx] = Molly.wrap_coords(sys.coords[idx] .- shift_vec, sys.boundary)
            Molly.apply_loggers!(sys, nothing, neighbors, step_n, run_loggers; n_threads=n_threads,
                           current_potential_energy=E_old, success=false,
                           energy_rate=(E_old / (sys.k * sim.temperature)))
        end
    end

    return sys
end