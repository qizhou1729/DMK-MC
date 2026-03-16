#include <hpdmk.h>
#include <tree.hpp>
#include <ewald.hpp>
#include <utils.hpp>

#include <mpi.h>
#include <omp.h>
#include <sctl.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace hpdmk;

namespace {
    struct AccuracyMetrics {
        double rel_l2 = 0.0;
        double rel_linf = 0.0;
        double net_force_norm = 0.0;
    };

    template <typename Real>
    void seeded_random_init(sctl::Vector<Real> &vec, const Real min, const Real max, const std::uint32_t seed) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<Real> distribution(min, max);
        for (int i = 0; i < vec.Dim(); ++i) {
            vec[i] = distribution(generator);
        }
    }

    AccuracyMetrics measure_force_accuracy(const int n_particles, const int digits, const int trial) {
        HPDMKParams params;
        params.n_per_leaf = 5;
        params.digits = digits;
        params.L = 20.0;
        params.init = DIRECT;

        sctl::Vector<double> r_src(3 * n_particles);
        sctl::Vector<double> charge(n_particles);

        const std::uint32_t seed_positions = 20260316u + 97u * static_cast<std::uint32_t>(n_particles) + 1009u * static_cast<std::uint32_t>(trial);
        const std::uint32_t seed_charge = 31415926u + 193u * static_cast<std::uint32_t>(n_particles) + 9176u * static_cast<std::uint32_t>(trial);
        seeded_random_init(r_src, 0.0, params.L, seed_positions);
        seeded_random_init(charge, -1.0, 1.0, seed_charge);
        unify_charge(charge);

        const double s = 4.0;
        const double alpha = 1.0;
        Ewald ewald(params.L, s, alpha, 1.0, &charge[0], &r_src[0], n_particles);
        const auto force_ref = ewald.compute_force();

        const sctl::Comm sctl_comm(MPI_COMM_WORLD);
        HPDMKPtTree<double> tree(sctl_comm, params, r_src, charge);
        tree.form_outgoing_pw();
        tree.form_incoming_pw();
        const auto force_hpdmk = tree.eval_force();

        double ref_norm2 = 0.0;
        double err_norm2 = 0.0;
        double ref_linf = 0.0;
        double err_linf = 0.0;
        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;

        for (long i = 0; i < force_hpdmk.Dim(); ++i) {
            const double ref_value = force_ref[i];
            const double err_value = force_hpdmk[i] - ref_value;
            ref_norm2 += ref_value * ref_value;
            err_norm2 += err_value * err_value;
            ref_linf = std::max(ref_linf, std::abs(ref_value));
            err_linf = std::max(err_linf, std::abs(err_value));

            if (i % 3 == 0) {
                fx += force_hpdmk[i];
            } else if (i % 3 == 1) {
                fy += force_hpdmk[i];
            } else {
                fz += force_hpdmk[i];
            }
        }

        AccuracyMetrics metrics;
        metrics.rel_l2 = std::sqrt(err_norm2 / std::max(ref_norm2, std::numeric_limits<double>::min()));
        metrics.rel_linf = err_linf / std::max(ref_linf, std::numeric_limits<double>::min());
        metrics.net_force_norm = std::sqrt(fx * fx + fy * fy + fz * fz);
        return metrics;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    omp_set_num_threads(1);

    const std::vector<int> particle_counts = {96, 128, 256, 512};
    const std::vector<int> digits_list = {3, 6, 9, 12};
    const int n_trials = 3;

    std::cout << "# HPDMK Force Accuracy Sweep\n\n";
    std::cout << "Comparison target: analytic Ewald bulk force\n";
    std::cout << "Parameters: L=20, n_per_leaf=5, init=DIRECT, s=4, alpha=1, OMP threads=1, trials=" << n_trials << "\n\n";
    std::cout << "| n_particles | digits | mean_rel_l2 | worst_rel_l2 | mean_rel_linf | worst_rel_linf | mean_net_force | achieved_digits |\n";
    std::cout << "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";

    for (const int n_particles : particle_counts) {
        for (const int digits : digits_list) {
            double sum_rel_l2 = 0.0;
            double sum_rel_linf = 0.0;
            double sum_net_force = 0.0;
            double worst_rel_l2 = 0.0;
            double worst_rel_linf = 0.0;

            for (int trial = 0; trial < n_trials; ++trial) {
                const auto metrics = measure_force_accuracy(n_particles, digits, trial);
                sum_rel_l2 += metrics.rel_l2;
                sum_rel_linf += metrics.rel_linf;
                sum_net_force += metrics.net_force_norm;
                worst_rel_l2 = std::max(worst_rel_l2, metrics.rel_l2);
                worst_rel_linf = std::max(worst_rel_linf, metrics.rel_linf);
            }

            const double mean_rel_l2 = sum_rel_l2 / n_trials;
            const double mean_rel_linf = sum_rel_linf / n_trials;
            const double mean_net_force = sum_net_force / n_trials;
            const double achieved_digits = -std::log10(std::max(mean_rel_l2, std::numeric_limits<double>::min()));

            std::cout << "| " << n_particles
                      << " | " << digits
                      << " | " << std::scientific << std::setprecision(3) << mean_rel_l2
                      << " | " << worst_rel_l2
                      << " | " << mean_rel_linf
                      << " | " << worst_rel_linf
                      << " | " << mean_net_force
                      << " | " << std::fixed << std::setprecision(2) << achieved_digits
                      << " |\n";
        }
    }

    MPI_Finalize();
    return 0;
}
