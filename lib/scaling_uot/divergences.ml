open Torch

type divergence = Tensor.t -> Tensor.t -> Tensor.t

let kl_divergence (p : Tensor.t) (q : Tensor.t) : Tensor.t =
  Tensor.(p * (log (p / q)) - p + q)

let tv_distance (p : Tensor.t) (q : Tensor.t) : Tensor.t =
  Tensor.(abs (p - q))

let range_constraint (alpha : float) (beta : float) (p : Tensor.t) (q : Tensor.t) : Tensor.t =
  let open Tensor in
  let condition = (p >= (f alpha * q)) * (p <= (f beta * q)) in
  where condition (zeros_like p) (ones_like p)

let proximal_kl (f : Tensor.t -> Tensor.t) (z : Tensor.t) (epsilon : float) : Tensor.t =
  let open Tensor in
  let rec iterate x iter =
    if iter > 100 then x  (* Max iterations to prevent infinite loop *)
    else
      let grad = (log (x / z) + f x / (f epsilon)) in
      let x_new = x * exp (neg grad) in
      if (((x_new - x) / x) |> abs |> max |> to_float0_exn) < 1e-8 then x_new
      else iterate x_new (iter + 1)
  in
  iterate z 0

let ghk_divergence (lambda : float) (p : Tensor.t) (q : Tensor.t) : Tensor.t =
  let open Tensor in
  f lambda * (p + q - f 2. * sqrt (p * q))