open Torch

type kernel = Tensor.t -> Tensor.t -> Tensor.t

exception Invalid_input of string

val gaussian_kernel : float -> kernel

val kernel_sgd_continuous_ot : 
  (unit -> Tensor.t) -> (unit -> Tensor.t) -> (Tensor.t -> Tensor.t -> float) -> 
  float -> int -> float -> kernel -> kernel -> float list * Tensor.t list * Tensor.t list

val adaptive_kernel_sgd_continuous_ot : 
  (unit -> Tensor.t) -> (unit -> Tensor.t) -> (Tensor.t -> Tensor.t -> float) -> 
  float -> int -> float -> kernel -> kernel -> float list * Tensor.t list * Tensor.t list