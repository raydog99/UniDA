open Torch
open Model

val train : 
  Model.t -> 
  Optimizer.t -> 
  (Tensor.t * Tensor.t) list -> 
  Tensor.t list -> 
  int -> 
  int -> 
  Model.t