open Torch

let generate_fragmented_hypercube (d : int) (k_star : int) (n : int) : Tensor.t * Tensor.t =
  let mu = Tensor.rand [n; d] |> Tensor.mul_scalar (Scalar.f 2.0) |> Tensor.sub_scalar (Scalar.f 1.0) in
  let t = Tensor.sign mu |> Tensor.narrow ~dim:1 ~start:0 ~length:k_star |> Tensor.sum ~dim:[1] |> Tensor.mul_scalar (Scalar.f 2.0) in
  let nu = Tensor.(mu + t.unsqueeze(-1) * (Tensor.ones [d] |> Tensor.narrow ~dim:0 ~start:0 ~length:k_star |> Tensor.unsqueeze(0))) in
  (mu, nu)