open Torch
open Model

val wasserstein_distance : 
  Model.DomainCritic.t -> Tensor.t -> Tensor.t -> Tensor.t

val gradient_penalty : 
  Model.DomainCritic.t -> Tensor.t -> Tensor.t

val cross_entropy_loss : 
  Tensor.t -> Tensor.t -> Tensor.topen Torch