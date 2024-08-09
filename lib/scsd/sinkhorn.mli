open Torch

type measure = {
  samples: Tensor.t;
  weights: Tensor.t;
}

val create_measure : Tensor.t -> Tensor.t -> measure
val uniform_measure : Tensor.t -> measure

val quadratic_cost : Tensor.t -> Tensor.t -> Tensor.t
val l1_cost : Tensor.t -> Tensor.t -> Tensor.t

val sinkhorn : measure -> measure -> Tensor.t -> float -> int -> Tensor.t * Tensor.t
val sinkhorn_divergence : measure -> measure -> Tensor.t -> float -> int -> Tensor.t

val sample_complexity_experiment : 
  int -> int -> float -> int -> (Tensor.t -> Tensor.t -> Tensor.t) -> float * float