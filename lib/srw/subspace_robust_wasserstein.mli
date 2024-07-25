open Torch

val displacement_matrix : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val cost_function : Tensor.t -> Tensor.t -> Tensor.t
val srw : Tensor.t -> Tensor.t -> int -> float -> int -> float -> Tensor.t
val srw_for_all_k : Tensor.t -> Tensor.t -> float -> int -> float -> (int * float) list
val choose_k_elbow : Tensor.t -> Tensor.t -> float -> int -> float -> float -> int