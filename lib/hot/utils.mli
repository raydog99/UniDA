open Types

val check_dataset_validity : dataset -> unit
val parallel_map : num_workers:int -> ('a -> 'b Lwt.t) -> 'a array -> 'b array
val frobenius_norm : Torch.Tensor.t -> float
val log : Logs.level -> string -> unit