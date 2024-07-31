open Torch

val load_cifar10_lt : 
  float -> 
  int -> 
  int -> 
  (Tensor.t * Tensor.t) list * Tensor.t list

val create_batches : 
  'a list -> 
  int -> 
  'a list list