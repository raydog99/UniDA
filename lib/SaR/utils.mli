open Torch

val balanced_accuracy : 
  Tensor.t -> 
  Tensor.t -> 
  int -> 
  float

val evaluate : 
  Model.t -> 
  (Tensor.t * Tensor.t) list -> 
  float