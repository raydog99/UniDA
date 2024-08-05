open Types
open Torch

val load_dataset : string -> dataset
val sample_batch : dataset -> int -> Tensor.t * Tensor.t