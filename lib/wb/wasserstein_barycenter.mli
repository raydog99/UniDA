open Torch
open Types

type t

val create : 
  Types.measure array -> 
  Types.cost_matrix -> 
  Types.incidence_matrix -> 
  float -> 
  t

val run : t -> Torch.Tensor.t * Torch.Tensor.t array * Torch.Tensor.t array
