\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xb264d9d7796a97f8
Maximize
  0.0225 d_xs[X0849362751] + 5000 d_xs[M5321647890] + 5000 d_xs[P7520836914]
   + 5000 d_xs[N5976283401] + 5000 d_xs[P7014836592]
   + 5000 d_xs[M3524691078] + 5000 d_xs[H1679043285]
   + 5000 d_xs[Y8495317260] - 362.2992753623189 d_xs[K5389142706]
   + 5000 d_xs[N5926738104] + 5000 d_xs[U1342708596]
   + 5000 d_xs[J1203854697] - p_f - 1.35 Constant
Subject To
 cons_veh_cap: 1.5 d_xs[X0849362751] + 2.6 d_xs[M5321647890]
   + 2.1 d_xs[P7520836914] + 2.4 d_xs[N5976283401] + 2.9 d_xs[P7014836592]
   + 2.9 d_xs[M3524691078] + 1.6 d_xs[H1679043285] + 1.7 d_xs[Y8495317260]
   + 1.2 d_xs[K5389142706] + 1.7 d_xs[N5926738104] + 1.4 d_xs[U1342708596]
   + 2.3 d_xs[J1203854697] <= 91
 cons_veh_min_load: 1.5 d_xs[X0849362751] + 2.6 d_xs[M5321647890]
   + 2.1 d_xs[P7520836914] + 2.4 d_xs[N5976283401] + 2.9 d_xs[P7014836592]
   + 2.9 d_xs[M3524691078] + 1.6 d_xs[H1679043285] + 1.7 d_xs[Y8495317260]
   + 1.2 d_xs[K5389142706] + 1.7 d_xs[N5926738104] + 1.4 d_xs[U1342708596]
   + 2.3 d_xs[J1203854697] + 99999 p_f >= 68
 cons_xs_bounds1[X0849362751]: d_xs[X0849362751] <= 78
 cons_xs_bounds1[M5321647890]: d_xs[M5321647890] <= 93
 cons_xs_bounds1[P7520836914]: d_xs[P7520836914] <= 86
 cons_xs_bounds1[N5976283401]: d_xs[N5976283401] <= 55
 cons_xs_bounds1[P7014836592]: d_xs[P7014836592] <= 35
 cons_xs_bounds1[M3524691078]: d_xs[M3524691078] <= 58
 cons_xs_bounds1[H1679043285]: d_xs[H1679043285] <= 33
 cons_xs_bounds1[Y8495317260]: d_xs[Y8495317260] <= 59
 cons_xs_bounds1[K5389142706]: d_xs[K5389142706] <= 69
 cons_xs_bounds1[N5926738104]: d_xs[N5926738104] <= 83
 cons_xs_bounds1[U1342708596]: d_xs[U1342708596] <= 94
 cons_xs_bounds1[J1203854697]: d_xs[J1203854697] <= 71
 cons_xs_bounds2[X0849362751]: d_xs[X0849362751] <= 332
 cons_xs_bounds2[M5321647890]: d_xs[M5321647890] <= 333
 cons_xs_bounds2[P7520836914]: d_xs[P7520836914] <= 259
 cons_xs_bounds2[N5976283401]: d_xs[N5976283401] <= 344
 cons_xs_bounds2[P7014836592]: d_xs[P7014836592] <= 265
 cons_xs_bounds2[M3524691078]: d_xs[M3524691078] <= 243
 cons_xs_bounds2[H1679043285]: d_xs[H1679043285] <= 341
 cons_xs_bounds2[Y8495317260]: d_xs[Y8495317260] <= 195
 cons_xs_bounds2[K5389142706]: d_xs[K5389142706] <= 297
 cons_xs_bounds2[N5926738104]: d_xs[N5926738104] <= 197
 cons_xs_bounds2[U1342708596]: d_xs[U1342708596] <= 164
 cons_xs_bounds2[J1203854697]: d_xs[J1203854697] <= 390
Bounds
 Constant = 1
Binaries
 p_f
Generals
 d_xs[X0849362751] d_xs[M5321647890] d_xs[P7520836914] d_xs[N5976283401]
 d_xs[P7014836592] d_xs[M3524691078] d_xs[H1679043285] d_xs[Y8495317260]
 d_xs[K5389142706] d_xs[N5926738104] d_xs[U1342708596] d_xs[J1203854697]
End
