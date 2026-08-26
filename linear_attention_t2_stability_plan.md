# Linear-Attention T2 Stability Implementation Plan

1. Add failing CPU tests for the smoothed feature-map derivative and its
   complement-denominator bound under adversarial Q/K channel ranges.
2. Add one configuration field for the required linear denominator floor;
   derive the minimum smoothing mass from the actual complement token count.
3. Thread that smoothing mass through every cube attention implementation:
   vendored autograd forward/backward, no-gradient guidance, fused JVP, and
   its per-tile state builder.  Keep the sparse branch unchanged.
4. Record the minimum linear denominator and maximum absolute T2 term during
   the JVP to distinguish the linear failure from sparse QK overflow.
5. Run targeted unit tests and a local short finite-loss/finite-gradient
   training smoke test.  Record settings, observations, and conclusions.
6. Commit and push the local change, synchronize the code to Nautilus-A100,
   then resume the 4-GPU run from `step_0006000.pt` under monitoring.
