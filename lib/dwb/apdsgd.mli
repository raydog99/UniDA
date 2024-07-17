type t

val create : Torch.Tensor.t -> Torch.Tensor.t -> Torch.Tensor.t -> Torch.Tensor.t -> Torch.Tensor.t -> float -> float -> float -> t
val run : t -> int -> t
val get_primal_result : t -> Torch.Tensor.t
val get_dual_result : t -> Torch.Tensor.t