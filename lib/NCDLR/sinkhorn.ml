open Torch
open Types
open Buffer

let sinkhorn_knopp (y : Tensor.t) (mu : Tensor.t) (w : Tensor.t) (max_iter : int) (epsilon : float) : Tensor.t =
  let y = Tensor.(div (exp (div_scalar y epsilon)) (sum (exp (div_scalar y epsilon)) ~dim:[1] ~keepdim:true)) in
  let rec loop y alpha beta iter =
    if iter >= max_iter then y
    else
      let alpha = Tensor.(div mu (mm y beta)) in
      let beta = Tensor.(div w (mm (transpose y ~dim0:0 ~dim1:1) alpha)) in
      let y_new = Tensor.(mul (mul y (unsqueeze alpha 1)) (unsqueeze beta 0)) in
      loop y_new alpha beta (iter + 1)
  in
  loop y (Tensor.ones_like mu) (Tensor.ones_like w) 0

let adaptive_self_labeling_loss (z_u : Tensor.t) (p_u : Tensor.t) (w : Tensor.t) (gamma : float) (buffer : t) : Tensor.t =
  let cost_matrix = Tensor.(neg (mm (transpose z_u ~dim0:0 ~dim1:1) p_u)) in
  Buffer.add buffer cost_matrix;
  let augmented_cost_matrix = Buffer.get buffer in
  let mu = Tensor.full [Tensor.shape augmented_cost_matrix |> List.hd] (1. /. float_of_int (Tensor.shape augmented_cost_matrix |> List.hd)) in
  let y = sinkhorn_knopp augmented_cost_matrix mu w 100 0.05 in
  let transport_cost = Tensor.(sum (mul augmented_cost_matrix y)) in
  let kl_div = Tensor.(sum (mul w (log (div w (Tensor.sum y ~dim:[0]))))) in
  Tensor.(add transport_cost (mul_scalar kl_div gamma))