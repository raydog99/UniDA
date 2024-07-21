open Torch

type dataset = Tensor.t

val pu_wasserstein : 
	dataset -> dataset -> float -> float -> Tensor.t

val pu_gromov_wasserstein : 
	dataset -> dataset -> float -> float -> Tensor.t

val create_pu_constraint : Tensor.t -> Tensor.t -> float -> Tensor.t

val classify : Tensor.t -> Tensor.t