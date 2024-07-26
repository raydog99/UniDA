open Torch

let sinkhorn_knopp cost_matrix epsilon max_iter =
  let m, n = Tensor.shape2_exn cost_matrix in
  let k = Tensor.exp Tensor.(neg cost_matrix / f epsilon) in
  let u = Tensor.ones [m; 1] in
  let v = Tensor.ones [1; n] in
  
  let rec iterate i u v =
    if i >= max_iter then (u, v)
    else
      let u_new = Tensor.div (Tensor.ones [m; 1]) Tensor.(matmul k v) in
      let v_new = Tensor.div (Tensor.ones [1; n]) Tensor.(matmul (transpose k ~dim0:0 ~dim1:1) u_new) in
      iterate (i + 1) u_new v_new
  in
  
  let u, v = iterate 0 u v in
  Tensor.(u * k * v)

let compute_optimal_transport cost_matrix =
  sinkhorn_knopp cost_matrix 0.01 100