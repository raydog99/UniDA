open Torch
open Types

val update_alpha : float -> float -> float -> float
val update_beta : float -> float -> float -> float
val compute_alpha : Tensor.t -> float -> float
val compute_beta : Tensor.t -> float -> float
val create_training_state : config -> training_state