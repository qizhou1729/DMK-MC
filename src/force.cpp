#include <hpdmk.h>
#include <tree.hpp>
#include <kernels.hpp>

#include <algorithm>
#include <cmath>
#include <complex>

namespace hpdmk {
    namespace {
        template <typename Real>
        sctl::Vector<Real> scatter_force_to_input_order(const sctl::Vector<Real> &force_sorted,
                                                        const sctl::Vector<Real> &indices_map_sorted) {
            const int n_particles = indices_map_sorted.Dim();
            sctl::Vector<Real> force_unsorted(3 * n_particles);
            force_unsorted.SetZero();

            for (int i_particle = 0; i_particle < n_particles; ++i_particle) {
                const auto unsorted_idx = static_cast<int>(indices_map_sorted[i_particle]);
                force_unsorted[3 * unsorted_idx] = force_sorted[3 * i_particle];
                force_unsorted[3 * unsorted_idx + 1] = force_sorted[3 * i_particle + 1];
                force_unsorted[3 * unsorted_idx + 2] = force_sorted[3 * i_particle + 2];
            }

            return force_unsorted;
        }

        template <typename Real>
        void rebuild_input_vectors(const HPDMKPtTree<Real> &tree,
                                   sctl::Vector<Real> &r_src,
                                   sctl::Vector<Real> &charge) {
            const int n_particles = tree.charge_sorted.Dim();
            r_src.ReInit(3 * n_particles);
            charge.ReInit(n_particles);

            for (int sorted_idx = 0; sorted_idx < n_particles; ++sorted_idx) {
                const int unsorted_idx = static_cast<int>(tree.indices_map_sorted[sorted_idx]);
                r_src[3 * unsorted_idx] = tree.r_src_sorted[3 * sorted_idx];
                r_src[3 * unsorted_idx + 1] = tree.r_src_sorted[3 * sorted_idx + 1];
                r_src[3 * unsorted_idx + 2] = tree.r_src_sorted[3 * sorted_idx + 2];
                charge[unsorted_idx] = tree.charge_sorted[sorted_idx];
            }
        }

        template <typename Real>
        sctl::Vector<Real> eval_force_terms(HPDMKPtTree<Real> &tree) {
            auto force = tree.eval_force_window();
            force += tree.eval_force_diff();
            force += tree.eval_force_res();
            return force;
        }
    }

    template <typename Real>
    sctl::Vector<Real> HPDMKPtTree<Real>::eval_force_window() {
        const int n_particles = charge_sorted.Dim();
        sctl::Vector<Real> force_sorted(3 * n_particles);
        force_sorted.SetZero();

        const Real inv_fourier_volume = 1 / std::pow(2 * M_PI, 3);
        const int d_window = 2 * n_window + 1;
        const int dims_window = d_window * d_window * (n_window + 1);
        const Real window_prefactor = inv_fourier_volume * std::pow(delta_k[0], 3);

        auto &window_root = outgoing_pw[root()];
        auto &window_weights = interaction_mat[0];

        for (int i_particle = 0; i_particle < n_particles; ++i_particle) {
            const Real x = r_src_sorted[i_particle * 3];
            const Real y = r_src_sorted[i_particle * 3 + 1];
            const Real z = r_src_sorted[i_particle * 3 + 2];
            const Real q = charge_sorted[i_particle];

            locate_particle(path_to_origin, x, y, z);
            form_outgoing_pw_single(outgoing_pw_origin, path_to_origin, x, y, z, q);

            auto &single_root = outgoing_pw_origin[0];
            for (int idx = 0; idx < dims_window; ++idx) {
                const int ix = idx % d_window;
                const int iy = (idx / d_window) % d_window;
                const int iz = idx / (d_window * d_window);

                const Real kx = mode_to_k(ix, n_window, delta_k[0]);
                const Real ky = mode_to_k(iy, n_window, delta_k[0]);
                const Real kz = mode_to_k(iz, n_window, delta_k[0]);
                const Real scale = window_prefactor * window_weights[idx] * std::imag(std::conj(single_root[idx]) * window_root[idx]);

                force_sorted[i_particle * 3] += scale * kx;
                force_sorted[i_particle * 3 + 1] += scale * ky;
                force_sorted[i_particle * 3 + 2] += scale * kz;
            }
        }

        return scatter_force_to_input_order(force_sorted, indices_map_sorted);
    }

    template <typename Real>
    sctl::Vector<Real> HPDMKPtTree<Real>::eval_force_diff() {
        const int n_particles = charge_sorted.Dim();
        sctl::Vector<Real> force_sorted(3 * n_particles);
        force_sorted.SetZero();

        const Real inv_fourier_volume = 1 / std::pow(2 * M_PI, 3);

        for (int i_particle = 0; i_particle < n_particles; ++i_particle) {
            const Real x = r_src_sorted[i_particle * 3];
            const Real y = r_src_sorted[i_particle * 3 + 1];
            const Real z = r_src_sorted[i_particle * 3 + 2];
            const Real q = charge_sorted[i_particle];

            locate_particle(path_to_origin, x, y, z);
            form_outgoing_pw_single(outgoing_pw_origin, path_to_origin, x, y, z, q);

            for (int l = 2; l < path_to_origin.Dim() - 1; ++l) {
                const sctl::Long i_node = path_to_origin[l];
                const int d_diff = 2 * n_diff + 1;
                const int dims_diff = d_diff * d_diff * (n_diff + 1);
                const Real diff_prefactor = inv_fourier_volume * std::pow(delta_k[l], 3);

                auto &single_level = outgoing_pw_origin[l];
                auto &incoming_level = incoming_pw[i_node];
                auto &diff_weights = interaction_mat[l];

                for (int idx = 0; idx < dims_diff; ++idx) {
                    const int ix = idx % d_diff;
                    const int iy = (idx / d_diff) % d_diff;
                    const int iz = idx / (d_diff * d_diff);

                    const Real kx = mode_to_k(ix, n_diff, delta_k[l]);
                    const Real ky = mode_to_k(iy, n_diff, delta_k[l]);
                    const Real kz = mode_to_k(iz, n_diff, delta_k[l]);
                    const Real scale = -diff_prefactor * diff_weights[idx] * std::imag(single_level[idx] * incoming_level[idx]);

                    force_sorted[i_particle * 3] += scale * kx;
                    force_sorted[i_particle * 3 + 1] += scale * ky;
                    force_sorted[i_particle * 3 + 2] += scale * kz;
                }
            }
        }

        return scatter_force_to_input_order(force_sorted, indices_map_sorted);
    }

    template <typename Real>
    sctl::Vector<Real> HPDMKPtTree<Real>::eval_force_res() {
        const int n_particles = charge_sorted.Dim();
        sctl::Vector<Real> force_sorted(3 * n_particles);
        force_sorted.SetZero();

        auto &node_attr = this->GetNodeAttr();
        for (int l = 2; l < max_depth; ++l) {
            const Real cutoff = boxsize[l];

            for (sctl::Long i_node : level_indices[l]) {
                if (!isleaf(node_attr[i_node]) || node_particles[i_node].Dim() == 0) {
                    continue;
                }

                for (int i = 0; i < node_particles[i_node].Dim() - 1; ++i) {
                    const int i_particle = node_particles[i_node][i];
                    const Real xi = r_src_sorted[i_particle * 3];
                    const Real yi = r_src_sorted[i_particle * 3 + 1];
                    const Real zi = r_src_sorted[i_particle * 3 + 2];

                    for (int j = i + 1; j < node_particles[i_node].Dim(); ++j) {
                        const int j_particle = node_particles[i_node][j];
                        const Real dx = r_src_sorted[j_particle * 3] - xi;
                        const Real dy = r_src_sorted[j_particle * 3 + 1] - yi;
                        const Real dz = r_src_sorted[j_particle * 3 + 2] - zi;
                        const Real r_ij = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (r_ij == 0 || r_ij > cutoff) {
                            continue;
                        }

                        const Real scale = charge_sorted[i_particle] * charge_sorted[j_particle] *
                            residual_kernel_derivative(r_ij, real_poly, cutoff) / r_ij;

                        force_sorted[i_particle * 3] += scale * dx;
                        force_sorted[i_particle * 3 + 1] += scale * dy;
                        force_sorted[i_particle * 3 + 2] += scale * dz;
                        force_sorted[j_particle * 3] -= scale * dx;
                        force_sorted[j_particle * 3 + 1] -= scale * dy;
                        force_sorted[j_particle * 3 + 2] -= scale * dz;
                    }
                }

                auto accumulate_cross_force = [&](const sctl::Long j_node) {
                    if (node_particles[j_node].Dim() == 0) {
                        return;
                    }

                    const auto shift_ij = node_shift(i_node, j_node);
                    const Real center_xi = centers[i_node * 3];
                    const Real center_yi = centers[i_node * 3 + 1];
                    const Real center_zi = centers[i_node * 3 + 2];
                    const Real center_xj = centers[j_node * 3];
                    const Real center_yj = centers[j_node * 3 + 1];
                    const Real center_zj = centers[j_node * 3 + 2];

                    for (auto i_particle : node_particles[i_node]) {
                        const Real xi = r_src_sorted[i_particle * 3] - center_xi - shift_ij[0];
                        const Real yi = r_src_sorted[i_particle * 3 + 1] - center_yi - shift_ij[1];
                        const Real zi = r_src_sorted[i_particle * 3 + 2] - center_zi - shift_ij[2];

                        for (auto j_particle : node_particles[j_node]) {
                            const Real xj = r_src_sorted[j_particle * 3] - center_xj;
                            const Real yj = r_src_sorted[j_particle * 3 + 1] - center_yj;
                            const Real zj = r_src_sorted[j_particle * 3 + 2] - center_zj;

                            const Real dx = xj - xi;
                            const Real dy = yj - yi;
                            const Real dz = zj - zi;
                            const Real r_ij = std::sqrt(dx * dx + dy * dy + dz * dz);
                            if (r_ij == 0 || r_ij > cutoff) {
                                continue;
                            }

                            const Real scale = Real(0.5) * charge_sorted[i_particle] * charge_sorted[j_particle] *
                                residual_kernel_derivative(r_ij, real_poly, cutoff) / r_ij;

                            force_sorted[i_particle * 3] += scale * dx;
                            force_sorted[i_particle * 3 + 1] += scale * dy;
                            force_sorted[i_particle * 3 + 2] += scale * dz;
                            force_sorted[j_particle * 3] -= scale * dx;
                            force_sorted[j_particle * 3 + 1] -= scale * dy;
                            force_sorted[j_particle * 3 + 2] -= scale * dz;
                        }
                    }
                };

                for (auto j_node : neighbors[i_node].coarsegrain) {
                    accumulate_cross_force(j_node);
                }
                for (auto j_node : neighbors[i_node].colleague) {
                    accumulate_cross_force(j_node);
                }
            }
        }

        return scatter_force_to_input_order(force_sorted, indices_map_sorted);
    }

    template <typename Real>
    sctl::Vector<Real> HPDMKPtTree<Real>::eval_force() {
        if (n_levels() <= 3) {
            return eval_force_terms(*this);
        }

        sctl::Vector<Real> r_src;
        sctl::Vector<Real> charge;
        rebuild_input_vectors(*this, r_src, charge);

        HPDMKParams force_params = params;
        force_params.n_per_leaf = std::max(force_params.n_per_leaf, 8);
        const int n_particles = charge.Dim();

        while (true) {
            const sctl::Comm sctl_comm(MPI_COMM_WORLD);
            HPDMKPtTree<Real> force_tree(sctl_comm, force_params, r_src, charge);
            if (force_tree.n_levels() <= 3 || force_params.n_per_leaf >= n_particles) {
                force_tree.form_outgoing_pw();
                force_tree.form_incoming_pw();
                return eval_force_terms(force_tree);
            }
            force_params.n_per_leaf = std::min(2 * force_params.n_per_leaf, n_particles);
        }
    }

    template sctl::Vector<float> HPDMKPtTree<float>::eval_force_window();
    template sctl::Vector<double> HPDMKPtTree<double>::eval_force_window();
    template sctl::Vector<float> HPDMKPtTree<float>::eval_force_diff();
    template sctl::Vector<double> HPDMKPtTree<double>::eval_force_diff();
    template sctl::Vector<float> HPDMKPtTree<float>::eval_force_res();
    template sctl::Vector<double> HPDMKPtTree<double>::eval_force_res();
    template sctl::Vector<float> HPDMKPtTree<float>::eval_force();
    template sctl::Vector<double> HPDMKPtTree<double>::eval_force();
}
