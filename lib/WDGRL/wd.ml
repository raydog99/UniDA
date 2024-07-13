open Torch

let wasserstein_distance critic source_features target_features =
  let source_critic = nn_apply critic source_features in
  let target_critic = nn_apply critic target_features in
  Tensor.(mean source_critic - mean target_critic)

let gradient_penalty critic features =
  let gradients = Tensor.grad_of_fn1 (nn_apply critic) features in
  let gradient_norm = Tensor.norm gradients ~p:2 ~dim:[1] ~keepdim:true in
  Tensor.((gradient_norm - scalar 1.) ** scalar 2.)

let cross_entropy_loss predictions labels =
  Tensor.nll_loss predictions labels