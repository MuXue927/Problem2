\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xcdc413f9a34ae44b
Maximize
  0.04 d_xs[W3421567089] + 0.04 d_xs[T2908743561] - p_f - Constant
Subject To
 cons_veh_cap: 1.8 d_xs[W3421567089] + 2.5 d_xs[T2908743561] <= 114
 cons_veh_min_load: 1.8 d_xs[W3421567089] + 2.5 d_xs[T2908743561]
   + 99999 p_f >= 65
 cons_xs_bounds1[W3421567089]: d_xs[W3421567089] <= 15
 cons_xs_bounds1[T2908743561]: d_xs[T2908743561] <= 19
 cons_xs_bounds2[W3421567089]: d_xs[W3421567089] <= 6
 cons_xs_bounds2[T2908743561]: d_xs[T2908743561] <= 19
Bounds
 Constant = 1
Binaries
 p_f
Generals
 d_xs[W3421567089] d_xs[T2908743561]
End
