open Torch

val check_input_dimensions :
  Tensor.t -> Tensor.t -> unit
(** Check if input tensors have valid dimensions for domain adaptation
    @param source_features Source domain features
    @param target_features Target domain features
    @raise Failure if dimensions are invalid
*)

val normalize_features :
  Tensor.t -> Tensor.t
(** Normalize features by subtracting mean and dividing by standard deviation
    @param features Input features tensor
    @return Normalized features tensor
*)

val create_random_features :
  int -> int -> Tensor.t
(** Create a random features tensor for testing
    @param num_samples Number of samples
    @param num_features Number of features per sample
    @return Random features tensor
*)