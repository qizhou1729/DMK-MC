
@testset "eval_energy" begin
    n_src = 100
    L = 20.0
    params = PDMK4MC.HPDMKParams(L = L, digits = 3, n_per_leaf = 5, init = PDMK4MC.DIRECT)

    r_src = rand(3, n_src) .* L
    charge = randn(n_src)
    charge .-= sum(charge) / n_src

    tree = PDMK4MC.create_tree(r_src, charge; params = params)
    PDMK4MC.form_outgoing_pw!(tree)
    PDMK4MC.form_incoming_pw!(tree)
    E_hpdmk = PDMK4MC.eval_energy(tree)

    # EwaldSummations
    boundary = Boundary((L, L, L), (1, 1, 1))
    atoms = Vector{Atom{Float64}}()
    for i in 1:n_src
        push!(atoms, Atom(type = 1, mass = 1.0, charge = charge[i]))
    end
    info = SimulationInfo(n_src, atoms, (0.0, L, 0.0, L, 0.0, L), boundary; min_r = 1.0, temp = 1.0)
    for i in 1:n_src
        info.particle_info[i].position = Point(r_src[1, i], r_src[2, i], r_src[3, i])
    end

    Ewald3D_interaction = Ewald3DInteraction(n_src, 3.0, 1.0, (L, L, L), ϵ = 1.0)

    neighbor = CellList3D(info, Ewald3D_interaction.r_c, boundary, 1)
    E_ewald = energy(Ewald3D_interaction, neighbor, info, atoms)

    @test isapprox(E_hpdmk / 4π, E_ewald, atol = 1e-3)
    destroy_tree!(tree)
end

@testset "eval_shift_energy" begin
    n_src = 100
    L = 20.0
    params = PDMK4MC.HPDMKParams(L = L, digits = 3, n_per_leaf = 5, init = PDMK4MC.DIRECT)

    r_src = rand(3, n_src) .* L
    charge = randn(n_src)
    charge .-= sum(charge) / n_src

    tree = PDMK4MC.create_tree(r_src, charge; params = params)
    PDMK4MC.form_outgoing_pw!(tree)
    PDMK4MC.form_incoming_pw!(tree)

    boundary = Boundary((L, L, L), (1, 1, 1))
    atoms = Vector{Atom{Float64}}()
    for i in 1:n_src
        push!(atoms, Atom(type = 1, mass = 1.0, charge = charge[i]))
    end
    info = SimulationInfo(n_src, atoms, (0.0, L, 0.0, L, 0.0, L), boundary; min_r = 1.0, temp = 1.0)
    for i in 1:n_src
        info.particle_info[i].position = Point(r_src[1, i], r_src[2, i], r_src[3, i])
    end

    Ewald3D_interaction = Ewald3DInteraction(n_src, 3.0, 1.0, (L, L, L), ϵ = 1.0)

    neighbor = CellList3D(info, Ewald3D_interaction.r_c, boundary, 1)
    E_ewald_old = energy(Ewald3D_interaction, neighbor, info, atoms)

    n_trials = 20
    for i in 1:n_trials
        dx = randn()
        dy = randn()
        dz = randn()
        i_particle = rand(1:n_src)

        E_hpdmk_shift = PDMK4MC.eval_shift_energy(tree, i_particle, dx, dy, dz)

        info_new = deepcopy(info)
        info_new.particle_info[i_particle].position = Point(mod(r_src[1, i_particle] + dx, L), mod(r_src[2, i_particle] + dy, L), mod(r_src[3, i_particle] + dz, L))

        neighbor_new = CellList3D(info_new, Ewald3D_interaction.r_c, boundary, 1)
        E_ewald_new = energy(Ewald3D_interaction, neighbor_new, info_new, atoms)
        E_ewald_shift = E_ewald_new - E_ewald_old
        @test isapprox(E_hpdmk_shift / 4π, E_ewald_shift, atol = 1e-3)
    end
    destroy_tree!(tree)
end

@testset "update_shift!" begin
    n_src = 100
    L = 20.0
    params = PDMK4MC.HPDMKParams(L = L, digits = 3, n_per_leaf = 5, init = PDMK4MC.DIRECT)

    r_src = rand(3, n_src) .* L
    charge = randn(n_src)
    charge .-= sum(charge) / n_src
   
    tree = PDMK4MC.create_tree(r_src, charge; params = params)
    PDMK4MC.form_outgoing_pw!(tree)
    PDMK4MC.form_incoming_pw!(tree)

    boundary = Boundary((L, L, L), (1, 1, 1))
    atoms = Vector{Atom{Float64}}()
    for i in 1:n_src
        push!(atoms, Atom(type = 1, mass = 1.0, charge = charge[i]))
    end
    info = SimulationInfo(n_src, atoms, (0.0, L, 0.0, L, 0.0, L), boundary; min_r = 1.0, temp = 1.0)
    for i in 1:n_src
        info.particle_info[i].position = Point(r_src[1, i], r_src[2, i], r_src[3, i])
    end

    Ewald3D_interaction = Ewald3DInteraction(n_src, 3.0, 1.0, (L, L, L), ϵ = 1.0)

    neighbor = CellList3D(info, Ewald3D_interaction.r_c, boundary, 1)
    E_ewald_old = energy(Ewald3D_interaction, neighbor, info, atoms)

    n_trials = 100
    for i in 1:n_trials
        dx = randn()
        dy = randn()
        dz = randn()
        i_particle = rand(1:n_src)

        E_hpdmk_shift = PDMK4MC.eval_shift_energy(tree, i_particle, dx, dy, dz)
        PDMK4MC.update_shift!(tree, i_particle, dx, dy, dz)

        position_old = info.particle_info[i_particle].position.coo
        info.particle_info[i_particle].position = Point(mod(position_old[1] + dx, L), mod(position_old[2] + dy, L), mod(position_old[3] + dz, L))
        neighbor = CellList3D(info, Ewald3D_interaction.r_c, boundary, 1)
        E_ewald_new = energy(Ewald3D_interaction, neighbor, info, atoms)
        E_ewald_shift = E_ewald_new - E_ewald_old
        @test isapprox(E_hpdmk_shift / 4π, E_ewald_shift, atol = 1e-3)
        E_ewald_old = E_ewald_new
    end
    destroy_tree!(tree)
end