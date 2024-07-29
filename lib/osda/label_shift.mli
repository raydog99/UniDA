open Torch

type label_shift_result = {
  pi: Tensor.t;
  nu: Tensor.t;
  predict_target_labels: unit -> Tensor.t;
}

val label_shift : ?epsilon:float -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> label_shift_result
(** Perform label shift correction *)

val get_class_proportions : label_shift_result -> Tensor.t
(** Get estimated class proportions *)