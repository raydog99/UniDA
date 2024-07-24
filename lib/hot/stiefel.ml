open Torch

let stiefel_alignment a =
  let u, _, v = Tensor.svd a ~some:true in
  Tensor.mm u (Tensor.transpose v 0 1)

let project_stiefel r =
  let u, _, v = Tensor.svd r ~some:true in
  Tensor.mm u (Tensor.transpose v 0 1)