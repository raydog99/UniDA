open Torch
open Utils

let sinkhorn cost_matrix epsilon n_iter =
  check_positive_float epsilon "epsilon";
  check_positive_float (float_of_int n_iter) "n_iter";
  let n, k = Tensor.shape cost_matrix in
  let m = Tensor.(exp (div_scalar cost_matrix (-.epsilon))) in
  let u = Tensor.ones [n] in
  let v = Tensor.ones [k] in
  let rec iterate i u v =
    if i = 0 then (u, v)
    else
      let u' = Tensor.(div u (sum (mul m v) ~dim:[1] ~keepdim:true)) in
      let v' = Tensor.(div v (sum (mul (transpose m ~dim0:0 ~dim1:1) u') ~dim:[0] ~keepdim:true)) in
      iterate (i - 1) u' v'
  in
  let u, v = iterate n_iter u v in
  Tensor.(mul (mul (unsqueeze u ~dim:1) m) (unsqueeze v ~dim:0))

let ot_loss embeddings centers epsilon n_iter =
  check_dimensions centers [Tensor.shape embeddings |> snd; Tensor.shape centers |> snd] "Invalid center dimensions";
  let cost_matrix = Tensor.cdist embeddings centers in
  let pi = sinkhorn cost_matrix epsilon n_iter in
  Tensor.sum (Tensor.mul cost_matrix pi)