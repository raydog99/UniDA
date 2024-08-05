open Types
open Torch

val encoder : config -> NN.t
val generate_prototypes : int -> int -> Tensor.t
val classify : Tensor.t -> Tensor.t -> Tensor.t
val mse_loss : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val parametric_cluster_size : Tensor.t -> int -> Tensor.t