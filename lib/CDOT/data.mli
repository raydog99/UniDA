open Torch

type dataset = {
  features: Tensor.t;
  labels: Tensor.t;
}

val create_rotating_moons : int -> float -> dataset

val create_source_target_seq : int -> int -> int -> float -> dataset * dataset list