open Torch

let compute_prototypes (features : Tensor.t) (labels : Tensor.t) (num_classes : int) : Tensor.t =
  let grouped_features = Tensor.group_by labels features in
  Tensor.mean grouped_features ~dim:[0]

let update_prototypes (old_prototypes : Tensor.t) (new_prototypes : Tensor.t) (momentum : float) : Tensor.t =
  Tensor.(old_prototypes * (Scalar.f (1. -. momentum)) + new_prototypes * (Scalar.f momentum))