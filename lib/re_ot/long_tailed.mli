open Torch
open Model
open Dataset

val create_long_tailed_ratio : int -> float -> Tensor.t
val create_teacher_based_ratio : Model.t -> Dataset.t -> Tensor.t