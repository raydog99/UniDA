open Torch

let stabilize_tensor (t : Tensor.t) (epsilon : float) : Tensor.t =
  Tensor.(max t (f epsilon))

let log_sum_exp (t : Tensor.t) : Tensor.t =
  let max_val = Tensor.max t in
  let shifted = Tensor.(t - max_val) in
  Tensor.(max_val + log (sum (exp shifted)))

let safe_div (a : Tensor.t) (b : Tensor.t) (epsilon : float) : Tensor.t =
  Tensor.(a / (b + f epsilon))