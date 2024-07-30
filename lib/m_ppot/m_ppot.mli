open Torch
open Types

val train_batch : training_state -> config -> Tensor.t -> Tensor.t -> Tensor.t -> loss
val train_epoch : training_state -> config -> Tensor.t -> Tensor.t -> Tensor.t -> unit
val train : config -> Tensor.t -> Tensor.t -> Tensor.t -> unit