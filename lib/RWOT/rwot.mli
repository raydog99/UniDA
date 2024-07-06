open Torch

module RWOT : sig
  type t

  val create : int -> int -> float -> t
  
  val calculate_class_centers : t -> Tensor.t -> Tensor.t -> unit
  
  val calculate_spatial_prototypical_matrix : Tensor.t -> Tensor.t -> Tensor.t
  
  val sharpen_probability_annotation_matrix : Tensor.t -> float -> Tensor.t
  
  val update_shrinking_subspace_reliability_cost_matrix : Tensor.t -> float -> Tensor.t
  
  val calculate_losses : t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t * Tensor.t
  
  val train : t -> Tensor.t * Tensor.t -> Tensor.t * Tensor.t -> int -> int -> int -> unit
end