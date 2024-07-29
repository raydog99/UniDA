open Torch

type osda_params = {
  epsilon: float;
  rejection_threshold: float;
}

val osda : Tensor.t -> Tensor.t -> Tensor.t -> osda_params -> 
           Rejection.rejection_result * Label_shift.label_shift_result * Tensor.t * Tensor.t * Tensor.t
(** Perform Open Set Domain Adaptation *)

val predict : Tensor.t -> Tensor.t -> Tensor.t -> osda_params -> Tensor.t * Tensor.t
(** Predict labels for target samples *)