open Torch

val sort_tensor : Tensor.t -> Tensor.t
val l_p_distance : float -> Tensor.t -> Tensor.t -> Tensor.t
val empirical_pdf : Tensor.t -> Tensor.t
val unpack_nn_params : Tensor.t -> int -> int -> int -> (Tensor.t list * Tensor.t list)

val linear : Tensor.t -> Tensor.t -> Tensor.t
val circular : Tensor.t -> Tensor.t -> Tensor.t
val polynomial : Tensor.t -> Tensor.t -> degree:int -> Tensor.t
val neural_network : Tensor.t -> Tensor.t -> num_layers:int -> hidden_dim:int -> Tensor.t