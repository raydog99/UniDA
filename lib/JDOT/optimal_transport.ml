open Torch

let sinkhorn_knopp cost_matrix epsilon max_iter =
  let m, n = Tensor.shape2_exn cost_matrix in
  let u = Tensor.ones [m] in
  let v = Tensor.ones [n] in
  
  let rec iterate i u v =
    if i >= max_iter then (u, v)
    else
      let k = Tensor.(exp (neg (div cost_matrix (f epsilon)))) in
      let u_new = Tensor.(div (f 1.) (mm k v)) in
      let v_new = Tensor.(div (f 1.) (mm (transpose k ~dim0:0 ~dim1:1) u_new)) in
      iterate (i + 1) u_new v_new
  in
  
  let u, v = iterate 0 u v in
  Tensor.(mul u (mm (exp (neg (div cost_matrix (f epsilon)))) v))

let compute_cost_matrix xs xt ys ft alpha =
  let d_xx = Tensor.cdist xs xt in
  let d_yy = Tensor.cdist ys ft in
  Tensor.(add (mul_scalar d_xx alpha) d_yy)