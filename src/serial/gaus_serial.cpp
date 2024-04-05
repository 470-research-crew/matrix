#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

// Custom timing macros (You'll need to define these based on your timing implementation)
#define START_TIMER(x)
#define STOP_TIMER(x)
#define GET_TIMER(x) 0.0 // Placeholder for timer value

// Uncomment this line to enable the alternative back substitution method
//#define USE_COLUMN_BACKSUB

// Use 64-bit IEEE arithmetic (change to float to use 32-bit arithmetic)
#define REAL double

class LinearSystemSolver {
public:
    int n;
    std::vector<REAL> A, x, b;
    bool debug_mode = false;
    bool triangular_mode = false;

    LinearSystemSolver() : n(0) {}

    void generateRandomSystem() {
        A.resize(n * n);
        b.resize(n);
        x.resize(n, 0);

        unsigned long seed = 0;
        for (int row = 0; row < n; ++row) {
            int colStart = triangular_mode ? row : 0;
            for (int col = colStart; col < n; ++col) {
                if (row != col) {
                    seed = (1103515245 * seed + 12345) % (1UL << 31);
                    A[row * n + col] = static_cast<REAL>(seed) / ULONG_MAX;
                } else {
                    A[row * n + col] = n / 10.0;
                }
            }
        }

        for (int row = 0; row < n; ++row) {
            b[row] = 0.0;
            for (int col = 0; col < n; ++col) {
                b[row] += A[row * n + col];
            }
        }
    }

    void readSystemFromFile(const std::string& filename) {
        std::FILE* fin = std::fopen(filename.c_str(), "r");
        if (!fin) {
            std::cerr << "Unable to open file \"" << filename << "\"\n";
            std::exit(EXIT_FAILURE);
        }
        if (fscanf(fin, "%d\n", &n) != 1) {
            std::cerr << "Invalid matrix file format\n";
            std::exit(EXIT_FAILURE);
        }

        A.resize(n * n);
        b.resize(n);
        x.resize(n, 0.0);

        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                std::fscanf(fin, "%lf", &A[row * n + col]);
            }
            std::fscanf(fin, "%lf", &b[row]);
        }
        std::fclose(fin);
    }

    void gaussianElimination() {
        for (int pivot = 0; pivot < n; ++pivot) {
            for (int row = pivot + 1; row < n; ++row) {
                REAL coeff = A[row * n + pivot] / A[pivot * n + pivot];
                A[row * n + pivot] = 0.0;
                for (int col = pivot + 1; col < n; ++col) {
                    A[row * n + col] -= A[pivot * n + col] * coeff;
                }
                b[row] -= b[pivot] * coeff;
            }
        }
    }

    void backSubstitution() {
        #ifndef USE_COLUMN_BACKSUB
        for (int row = n - 1; row >= 0; --row) {
            REAL tmp = b[row];
            for (int col = row + 1; col < n; ++col) {
                tmp -= A[row * n + col] * x[col];
            }
            x[row] = tmp / A[row * n + row];
        }
        #else
        // Column-oriented back substitution code would go here
        #endif
    }

    REAL findMaxError() const {
        REAL maxError = 0.0;
        for (int i = 0; i < n; ++i) {
            REAL error = std::fabs(x[i] - 1.0);
            if (error > maxError) maxError = error;
        }
        return maxError;
    }

    void printMatrix(const std::vector<REAL>& mat, int rows, int cols) const {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout << mat[row * cols + col] << " ";
        }
        std::cout << std::endl;
    }
}

};

int main(int argc, char* argv[]) {
    LinearSystemSolver solver;
    int option;
    while ((option = getopt(argc, argv, "dt")) != -1) {
        switch (option) {
        case 'd':
            solver.debug_mode = true;
            break;
        case 't':
            solver.triangular_mode = true;
            break;
        default:
            std::cout << "Usage: " << argv[0] << " [-dt] <file|size>\n";
            return EXIT_FAILURE;
        }
    }
    if (optind != argc - 1) {
        std::cout << "Usage: " << argv[0] << " [-dt] <file|size>\n";
        return EXIT_FAILURE;
    }

    START_TIMER(init);
    std::string arg = argv[optind];
    if (arg.find_first_not_of("0123456789") == std::string::npos) {
        // Argument is a size
        solver.n = std::stoi(arg);
        solver.generateRandomSystem();
    } else {
        // Argument is a filename
        solver.readSystemFromFile(arg);
    }
    STOP_TIMER(init);

    if (solver.debug_mode) {
        std::cout << "Original A = \n";
        solver.printMatrix(solver.A, solver.n, solver.n);
        std::cout << "Original b = \n";
        solver.printMatrix(solver.b, solver.n, 1);
    }

    START_TIMER(gaus);
    if (!solver.triangular_mode) {
        solver.gaussianElimination();
    }
    STOP_TIMER(gaus);

    START_TIMER(bsub);
    solver.backSubstitution();
    STOP_TIMER(bsub);

    if (solver.debug_mode) {
        std::cout << "Triangular A = \n";
        solver.printMatrix(solver.A, solver.n, solver.n);
        std::cout << "Updated b = \n";
        solver.printMatrix(solver.b, solver.n, 1);
        std::cout << "Solution x = \n";
        solver.printMatrix(solver.x, solver.n, 1);
    }

    std::cout << "Nthreads=1  ERR=" << solver.findMaxError()
              << "  INIT: " << GET_TIMER(init) << "s  GAUS: " << GET_TIMER(gaus) << "s  BSUB: " << GET_TIMER(bsub) << "s\n";

    return EXIT_SUCCESS;
}
