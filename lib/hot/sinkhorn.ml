open Torch
open Types

let sinkhorn epsilon cost_matrix max_iter tol =
  let m, n = Tensor.shape cost_matrix in
  let k = Tensor.exp (Tensor.div_scalar cost_matrix (-. epsilon)) in
  let v = Tensor.ones [n] in
  
  let rec iterate u v iter =
    if iter >= max_iter then (u, v)
    else
      let u' = Tensor.div (Tensor.ones [m]) (Tensor.mm k v) in
      let v' = Tensor.div (Tensor.ones [n]) (Tensor.mm (Tensor.transpose k 0 1) u') in
      let diff = Tensor.sub u' u |> Utils.frobenius_norm in
      if diff < tol then (u', v')
      else iterate u' v' (iter + 1)
  in
  
  let u, v = iterate (Tensor.ones [m]) v 0 in
  Tensor.mul (Tensor.mul (Tensor.diag u) k) (Tensor.diag v)

let entropy p =
  let p_log = Tensor.log p in
  Tensor.sum (Tensor.mul p p_log) |> Tensor.item