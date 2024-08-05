open Types
open Torch

val train : config -> NN.t -> Tensor.t -> dataset -> dataset -> NN.t