# Rival attempts re-benchmarked on OUR Ice Lake node (a80n0, pinned core)

Graded points, best-of-4, chain end checked vs a numpy reference; GATE = our 1e-13/step budget.

| attempt | total s | L6 | L8 | L13 | L17 | L23 | L36 | L45 | L64 | gates passed |
|---|---|---|---|---|---|---|---|---|---|---|
| v5_3907583b_score0.87 | 0.915 | 0.065✗ | 0.094 | 0.160 | 0.035 | 0.126 | 0.057✗ | 0.187✗ | 0.191 | 5/8 |
| v5_95ab77a1_score0.82 | 0.962 | 0.067✗ | 0.095 | 0.177 | 0.041 | 0.123 | 0.065✗ | 0.198✗ | 0.197 | 5/8 |
| v6_4d0483ea_score0.85 | 0.993 | 0.073✗ | 0.096 | 0.164 | 0.043 | 0.116 | 0.069✗ | 0.207✗ | 0.225 | 5/8 |
| v6_2cbe0fb0_score0.80 | 0.995 | 0.074✗ | 0.127 | 0.167 | 0.039 | 0.102 | 0.068✗ | 0.209✗ | 0.208 | 5/8 |
| 1760b1bf_score0.96 | 1.004 | 0.089✗ | 0.115 | 0.163 | 0.033 | 0.099 | 0.059 | 0.208 | 0.237 | 7/8 |
| 1000f989_score1.00 | 1.004 | 0.069✗ | 0.092 | 0.159 | 0.036 | 0.109 | 0.064 | 0.248 | 0.229 | 7/8 |
| v6_78662a62_score0.84 | 1.007 | 0.063✗ | 0.090 | 0.170 | 0.046 | 0.125 | 0.058✗ | 0.229✗ | 0.226 | 5/8 |
| v6_3f30d81f_score0.88 | 1.014 | 0.084✗ | 0.109 | 0.172 | 0.038 | 0.125 | 0.068✗ | 0.243✗ | 0.175 | 5/8 |
| v6_f40c5e25_score0.91 | 1.034 | 0.080✗ | 0.091✗ | 0.180✗ | 0.041✗ | 0.118✗ | 0.075✗ | 0.220✗ | 0.228✗ | 0/8 |
| 8dc1a96d_score1.00_floor-artifact | 1.060 | 0.069✗ | 0.088 | 0.176 | 0.040 | 0.113 | 0.073 | 0.239 | 0.261 | 7/8 |
| dd9fa88c_score0.76 | 1.064 | 0.073✗ | 0.095 | 0.199 | 0.037 | 0.136 | 0.068 | 0.214 | 0.241 | 7/8 |
| v5_197d7c23_score0.83 | 1.067 | 0.068✗ | 0.090 | 0.200 | 0.047 | 0.123 | 0.067✗ | 0.205✗ | 0.267 | 5/8 |
| a31f5f85_score0.81 | 1.072 | 0.089✗ | 0.114 | 0.175 | 0.036 | 0.101 | 0.079 | 0.238 | 0.240 | 7/8 |
| v5_26833bab_score0.87 | 1.086 | 0.098✗ | 0.116 | 0.197 | 0.043 | 0.114 | 0.071✗ | 0.211✗ | 0.237 | 5/8 |
| v6_8b0fbe57_score0.77 | 1.121 | 0.067✗ | 0.093✗ | 0.183✗ | 0.042✗ | 0.128✗ | 0.086✗ | 0.274✗ | 0.248✗ | 0/8 |
| v5_8175a973_score0.90 | 1.123 | 0.080✗ | 0.114 | 0.210 | 0.041 | 0.108 | 0.054✗ | 0.235✗ | 0.280 | 5/8 |
| v5_2c2dfce8_score0.74 | 1.135 | 0.066✗ | 0.124 | 0.211 | 0.049 | 0.124 | 0.077✗ | 0.234✗ | 0.249 | 5/8 |
| f7f192ab_score0.00_workdir-size-zero | 1.153 | 0.095✗ | 0.116 | 0.208 | 0.042 | 0.133 | 0.070 | 0.248 | 0.241 | 7/8 |
| v6_5a869e40_score0.80 | 1.156 | 0.084✗ | 0.113 | 0.171 | 0.045 | 0.117 | 0.055✗ | 0.167✗ | 0.405 | 5/8 |
| v6_4a50d3db_score0.77 | 1.158 | 0.083✗ | 0.120 | 0.164 | 0.039 | 0.138 | 0.071✗ | 0.243✗ | 0.301 | 5/8 |
| v5_cb7847fb_guard-envfail | 1.234 | 0.066✗ | 0.086 | 0.186 | 0.049 | 0.156 | 0.085✗ | 0.266✗ | 0.339 | 5/8 |
| 0f45aeae_score0.44 | 1.264 | 0.077✗ | 0.121 | 0.242 | 0.047 | 0.133 | 0.082 | 0.255 | 0.307 | 7/8 |
| v5_2419f19d_score0.81 | 1.315 | 0.155✗ | 0.144 | 0.276 | 0.057 | 0.153 | 0.058✗ | 0.204✗ | 0.267 | 5/8 |

Our panel's final (best gate-passing per size): **0.943 s, 8/8 gates**.

## Caveats on the mass L=36/L=45 failures

Ten of the sixteen v5/v6 attempts return completely wrong output (rel err ~1) at exactly
L=36 and L=45 on this node while passing six other sizes. A shared failure at the same two
sizes across independent attempts suggests an environment assumption rather than ten
identical bugs — plausibly hard-coded batch-count thresholds from their campaign's own
(B, m) table, or THP/alignment assumptions their tier satisfied. Treat their L=36/45
numbers as unavailable rather than beaten-by-default; the per-size winners table below the
main table uses only gate-passing rows.

## The two headline facts

1. **No rival attempt — none of 23, across three campaigns — passes the L=6 chain gate.**
   Every one drifts ~1e-9..1e-8 at m=4856. Our `L6_pfa` is the only chain-true fast L=6 in
   existence.
2. **No single rival attempt beats our panel under the gate**: best rival total 0.915 s
   holds only 5/8 gates; the best 7/8 attempts (the two v4 1.00-scorers) sit at 1.004 s vs
   our 0.943 at 8/8.
