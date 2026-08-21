/* Minimal FFTW3-MPI 3D c2c check: slab-decomposed forward+backward roundtrip. */
#include <complex.h>
#include <fftw3-mpi.h>
#include <math.h>
#include <stdio.h>
int main(int argc, char **argv) {
    const ptrdiff_t L = 32;
    MPI_Init(&argc, &argv);
    fftw_mpi_init();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ptrdiff_t local_n0, local_0_start;
    ptrdiff_t alloc = fftw_mpi_local_size_3d(L, L, L, MPI_COMM_WORLD, &local_n0, &local_0_start);
    fftw_complex *d = fftw_alloc_complex(alloc);
    fftw_plan fwd = fftw_mpi_plan_dft_3d(L, L, L, d, d, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan bwd = fftw_mpi_plan_dft_3d(L, L, L, d, d, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    for (ptrdiff_t i = 0; i < local_n0 * L * L; ++i)
        d[i] = (double)((local_0_start * L * L + i) % 17) + 0.5 * I;
    fftw_complex *save = fftw_alloc_complex(alloc);
    for (ptrdiff_t i = 0; i < local_n0 * L * L; ++i) save[i] = d[i];
    fftw_execute(fwd);
    fftw_execute(bwd);
    double err = 0.0, nrm = 0.0;
    for (ptrdiff_t i = 0; i < local_n0 * L * L; ++i) {
        fftw_complex diff = d[i] / (double)(L * L * L) - save[i];
        err += creal(diff) * creal(diff) + cimag(diff) * cimag(diff);
        nrm += creal(save[i]) * creal(save[i]) + cimag(save[i]) * cimag(save[i]);
    }
    double gerr = 0, gnrm = 0;
    MPI_Reduce(&err, &gerr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nrm, &gnrm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("FFTW-MPI %d ranks, L=%td slabs: roundtrip rel err = %.3e\n",
               size, L, sqrt(gerr / gnrm));
    fftw_destroy_plan(fwd); fftw_destroy_plan(bwd);
    fftw_free(d); fftw_free(save);
    fftw_mpi_cleanup(); MPI_Finalize();
    return 0;
}
