open Torch
open Types

val m_ppot_loss : Tensor.t -> Tensor.t -> float -> Tensor.t
val reweighted_entropy_loss : Tensor.t -> Tensor.t -> Tensor.t
val reweighted_cross_entropy_loss : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val compute_loss : Tensor.t -> Tensor.t -> Tensor.t -> config -> loss