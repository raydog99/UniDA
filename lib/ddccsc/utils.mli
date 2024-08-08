type error =
  | InvalidDimensions of string
  | InvalidParameter of string

exception ClusteringError of error

val check_dimensions : Torch.Tensor.t -> int list -> string -> unit
val check_positive_float : float -> string -> unit
val last : 'a list -> 'a
val string_of_list : ('a -> string) -> 'a list -> string