/*
 * par_gauss.cpp
 *
 * CS 470 Project 2 (OpenMP)
 * OpenMP parallelized version in C++
 *
 * by: Lexi Krobath and Mark Myers
 *
 * Compile with g++ and enable OpenMP, e.g., g++ -fopenmp par_gauss.cpp -o par_gauss
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <fstream>
#include <limits>
#include <getopt.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "timer.h" // Assuming the same timer utility is used, otherwise need to adapt for C++

#define USE_COLUMN_BACKSUB
#define REAL double

class LinearSystem {
public:
    int n;
    std::vector<REAL> A, x, b;
    bool debug_mode = false;
    bool triangular_mode = false;

    LinearSystem() : n(0) {}

    void rand_system() {
        A.resize(n * n);
        b.resize(n);
        x.resize(n, 0);

        unsigned long seed = 0;
#pragma omp parallel for default(none) shared(n, A, triangular_mode, seed)
        for (int row = 0; row < n; row++) {
            int col = triangular_mode ? row : 0;
            for (; col < n; col++) {
                if (row != col) {
                    seed = (1103515245 * seed + 12345) % (1 << 31);
                    A[row * n + col] = static_cast<REAL>(seed) / static_cast<REAL>(ULONG_MAX);
                } else {
                    A[row * n + col] = n / 10.0;
                }
            }
        }

#pragma omp parallel for default(none) shared(n, A, b)
        for (int row = 0; row < n; row++) {
            b[row] = 0.0;
            for (int col = 0; col < n; col++) {
                b[row] += A[row * n + col];
            }
        }
    }

    void read_system(const std::string &fn) {
        std::ifstream fin(fn);
        if (!fin) {
            throw std::runtime_error("Unable to open file \"" + fn + "\"");
        }

        fin >> n;
        if (!fin) {
            throw std::runtime_error("Invalid matrix file format");
        }

        A.resize(n * n);
        b.resize(n);
        x.resize(n, 0);

        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                fin >> A[row * n + col];
            }
            fin >> b[row];
        }
    }

    void gaussian_elimination() {
        for (int pivot = 0; pivot < n; pivot++) {
#pragma omp parallel for default(none) shared(A, n, b, pivot)
            for (int row = pivot + 1; row < n; row++) {
                REAL coeff = A[row * n + pivot] / A[pivot * n + pivot];
                A[row * n + pivot] = 0.0;
                for (int col = pivot + 1; col < n; col++) {
                    A[row * n + col] -= A[pivot * n + col] * coeff;
                }
                b[row] -= b[pivot] * coeff;
            }
        }
    }

    void back_substitution_row() {
        for (int row = n - 1; row >= 0; row--) {
            REAL tmp = b[row];
#pragma omp parallel for default(none) shared(A, x, n, row) reduction(- : tmp)
            for (int col = row + 1; col < n; col++) {
                tmp -= A[row * n + col] * x[col];
            }
            x[row] = tmp / A[row * n + row];
        }
    }

    void back_substitution_column() {
#pragma omp parallel for default(none) shared(n, b, x)
        for (int row = 0;
        row < n; row++) {
            x[row] = b[row];
        }
        for (int col = n - 1; col >= 0; col--) {
            x[col] /= A[col * n + col];
#pragma omp parallel for default(none) shared(A, x, n, col)
            for (int row = 0; row < col; row++) {
                x[row] -= A[row * n + col] * x[col];
            }
        }
    }

    REAL find_max_error() {
        REAL error = 0.0;
        for (int row = 0; row < n; row++) {
            REAL tmp = std::fabs(x[row] - 1.0);
            if (tmp > error) {
                error = tmp;
            }
        }
        return error;
    }

    void print_matrix(const std::vector<REAL>& mat, int rows, int cols) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                std::cout << std::scientific << mat[row * cols + col] << " ";
            }
            std::cout << "\n";
        }
    }
};

int main(int argc, char *argv[]) {
    LinearSystem ls;
    int opt;
    while ((opt = getopt(argc, argv, "dt")) != -1) {
        switch (opt) {
            case 'd':
                ls.debug_mode = true;
                break;
            case 't':
                ls.triangular_mode = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-dt] <file|size>\n";
                return EXIT_FAILURE;
        }
    }
    if (optind != argc - 1) {
        std::cerr << "Usage: " << argv[0] << " [-dt] <file|size>\n";
        return EXIT_FAILURE;
    }

    START_TIMER(init)
    try {
        long size = std::strtol(argv[optind], nullptr, 10);
        if (size == 0) {
            ls.read_system(argv[optind]);
        } else {
            ls.n = static_cast<int>(size);
            ls.rand_system();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    STOP_TIMER(init)

    if (ls.debug_mode) {
        std::cout << "Original A = \n";
        ls.print_matrix(ls.A, ls.n, ls.n);
        std::cout << "Original b = \n";
        ls.print_matrix(ls.b, ls.n, 1);
    }

    START_TIMER(gaus)
    if (!ls.triangular_mode) {
        ls.gaussian_elimination();
    }
    STOP_TIMER(gaus)

    START_TIMER(bsub)
#ifndef USE_COLUMN_BACKSUB
    ls.back_substitution_row();
#else
    ls.back_substitution_column();
#endif
    STOP_TIMER(bsub)

    if (ls.debug_mode) {
        std::cout << "Triangular A = \n";
        ls.print_matrix(ls.A, ls.n, ls.n);
        std::cout << "Updated b = \n";
        ls.print_matrix(ls.b, ls.n, 1);
        std::cout << "Solution x = \n";
        ls.print_matrix(ls.x, ls.n, 1);
    }

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    std::cout << "Nthreads=" << threads << "  ERR=" << std::scientific << ls.find_max_error() << "  INIT: " << GET_TIMER(init) << "s  GAUS: " << GET_TIMER(gaus) << "s  BSUB: " << GET_TIMER(bsub) << "s\n";

    return EXIT_SUCCESS;
}
