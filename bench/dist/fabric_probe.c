/* Minimal MPI ping-pong + all-to-all probe, run before the FFT sweep so every result carries
 * a direct measurement of the fabric it was taken on.
 *
 * This exists because of a real misread waiting to happen: on the devel/prod nodes the same
 * 32-rank 128^3 transpose took 0.006 s inside one node and 0.645 s across two -- 107x.  Read
 * off the FFT alone, that looks like "communication dominates at two nodes", which is the
 * survey's headline and therefore the easy thing to believe.  It is not: it is the transport.
 * Measuring latency and bandwidth directly separates "the network is slow here" from "the
 * transpose is expensive", and the two demand completely different conclusions.
 *
 * Prints one line per size, plus a summary the harness greps into environment.txt.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double now(void) { return MPI_Wtime(); }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[MPI_MAX_PROCESSOR_NAME];
    int hlen = 0;
    MPI_Get_processor_name(host, &hlen);

    /* Report the rank-to-host mapping, so a "2 node" run that silently landed on one node
     * is visible rather than assumed. */
    if (rank == 0) {
        char *all = malloc((size_t)size * MPI_MAX_PROCESSOR_NAME);
        MPI_Gather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                   all, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
        printf("probe: %d ranks; rank0=%s rank%d=%s\n", size, all,
               size - 1, all + (size_t)(size - 1) * MPI_MAX_PROCESSOR_NAME);
        free(all);
    } else {
        MPI_Gather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                   NULL, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    /* Ping-pong between rank 0 and the LAST rank.  With ppr:<n>:node mapping the last rank is
     * on the far node, so this is the inter-node path -- the one that matters. */
    int peer = size - 1;
    size_t sizes[] = {8, 1024, 65536, 1048576, 8388608};
    int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    double lat_us = -1.0, bw_best = -1.0;

    if (size >= 2) {
        char *buf = malloc(sizes[nsizes - 1]);
        memset(buf, 1, sizes[nsizes - 1]);
        for (int s = 0; s < nsizes; s++) {
            size_t n = sizes[s];
            int reps = n < 65536 ? 200 : (n < 1048576 ? 50 : 10);
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = now();
            if (rank == 0) {
                for (int r = 0; r < reps; r++) {
                    MPI_Send(buf, (int)n, MPI_CHAR, peer, 0, MPI_COMM_WORLD);
                    MPI_Recv(buf, (int)n, MPI_CHAR, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            } else if (rank == peer) {
                for (int r = 0; r < reps; r++) {
                    MPI_Recv(buf, (int)n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buf, (int)n, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
                }
            }
            double t1 = now();
            if (rank == 0) {
                double half = (t1 - t0) / (2.0 * reps);      /* one-way */
                double bw = (double)n / half / 1e9;          /* GB/s */
                printf("probe: pingpong %9zu B  one-way %10.3f us  %8.3f GB/s\n",
                       n, half * 1e6, bw);
                if (s == 0) lat_us = half * 1e6;
                if (bw > bw_best) bw_best = bw;
            }
        }
        free(buf);
    }

    /* An all-to-all of the size an FFT transpose actually moves: this is the primitive the
     * reshape stage is built from, so its rate bounds the transpose no matter what heFFTe does. */
    size_t per_peer = 262144;                     /* 256 KB to each rank */
    char *sbuf = malloc(per_peer * (size_t)size);
    char *rbuf = malloc(per_peer * (size_t)size);
    memset(sbuf, 2, per_peer * (size_t)size);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = now();
    for (int r = 0; r < 5; r++)
        MPI_Alltoall(sbuf, (int)per_peer, MPI_CHAR, rbuf, (int)per_peer, MPI_CHAR, MPI_COMM_WORLD);
    double t1 = now();
    if (rank == 0) {
        double per = (t1 - t0) / 5.0;
        double moved = (double)per_peer * (double)size * (double)(size - 1) / (double)size;
        printf("probe: alltoall %zu B/peer x %d ranks  %10.3f ms  %8.3f GB/s/rank\n",
               per_peer, size, per * 1e3, moved / per / 1e9);
        printf("probe-summary: latency_us=%.3f peak_bw_GBs=%.3f alltoall_ms=%.3f\n",
               lat_us, bw_best, per * 1e3);
    }
    free(sbuf);
    free(rbuf);
    MPI_Finalize();
    return 0;
}
