open Torch

let wfr_cost (d : Tensor.t) (lambda : float) : Tensor.t =
  let open Tensor in
  let cos_plus = fun x -> max x (f (Float.pi /. 2.)) |> cos in
  f lambda * (log (cos_plus (d / f lambda)))

let wfr_proximal (mu : Tensor.t) (nu : Tensor.t) (epsilon : float) : Tensor.t =
  let open Tensor in
  let c = mu / nu in
  let lambda = f epsilon /. 2. in
  let a = f 1. + c * (f 1. / lambda) in
  let b = sqrt (sq a - f 4. * c * (f 1. / lambda)) in
  nu * ((a - b) / f 2.)

let wfr_distance (mu : Tensor.t) (nu : Tensor.t) (pi : Tensor.t) (c : Tensor.t) (epsilon : float) : float =
  let open Tensor in
  let transport_cost = sum (c * pi) in
  let marginal_cost = f epsilon * (sum (mu * log (mu / (sum1 pi))) + sum (nu * log (nu / (sum1' pi)))) in
  (transport_cost + marginal_cost) |> to_float0_exn