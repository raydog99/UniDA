open Torch
open Model
open Dataset

val compute_cost_matrix : Tensor.t -> Tensor.t -> Tensor.t
val relative_entropy_regularization : Tensor.t -> Tensor.t -> Tensor.t
val re_ot_optimize : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> float -> int -> Tensor.t
val inverse_re_ot : Tensor.t -> Tensor.t -> Tensor.t -> float -> int -> Tensor.t
val re_ot_loss : Tensor.t -> Tensor.t -> Tensor.t
val train : Model.t -> Dataset.t -> int -> float -> int -> float -> int -> unit
val infer : Model.t -> Tensor.t -> Tensor.t
val update_q : Tensor.t -> Tensor.t -> Tensor.t -> int -> int -> Tensor.t