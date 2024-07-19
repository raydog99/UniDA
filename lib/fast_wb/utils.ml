open Torch

let pairwise_distances x y p =
  let x_sqnorms = Tensor.(
    pow_scalar x p
    |> sum ~dim:[-1] ~keepdim:true
  ) in
  let y_sqnorms = Tensor.(
    pow_scalar y p
    |> sum ~dim:[-1] ~keepdim:true
    |> transpose ~dim0:(-2) ~dim1:(-1)
  ) in
  let xy = Tensor.(matmul x (transpose y ~dim0:(-2) ~dim1:(-1))) in
  Tensor.(
    add (add x_sqnorms y_sqnorms) (neg (mul_scalar xy 2.))
    |> relu
    |> pow_scalar (1. /. p)
  )

let kl_projection x =
  let x_sum = Tensor.sum x in
  let x_log = Tensor.(log (div x x_sum)) in
  let x_entropy = Tensor.(neg (sum (mul x x_log))) in
  let threshold = Tensor.log (Tensor.of_float (float (Tensor.shape2_exn x).(0))) in
  if Tensor.(to_float0_exn (lt x_entropy threshold)) then
    Tensor.(exp (sub x_log (sub x_entropy threshold)))
  else
    Tensor.(div x x_sum)