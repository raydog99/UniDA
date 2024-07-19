open Torch

module type S = sig
  val sinkhorn :
    ?iterations:int ->
    ?epsilon:float ->
    cost:Tensor.t ->
    a:Tensor.t ->
    b:Tensor.t ->
    unit ->
    Tensor.t * Tensor.t * Tensor.t
end

module Make () : S