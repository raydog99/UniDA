open Torch

module type WassersteinBarycenter = sig
  val compute_barycenter :
    ?iterations:int ->
    ?t0:float ->
    ?lambda:float ->
    supports:Tensor.t list ->
    weights:Tensor.t list ->
    p:float ->
    init_support:Tensor.t ->
    theta:Tensor.t ->
    unit ->
    Tensor.t

  val proximal_update :
    a:Tensor.t -> b:Tensor.t -> t0:float -> alpha:Tensor.t -> Tensor.t
end

module Make (S : Sinkhorn.S) : WassersteinBarycenter