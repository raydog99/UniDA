type t = {
  alpha : float;
  epsilon : float;
  max_iter : int;
  learning_rate : float;
  batch_size : int;
  num_iterations : int;
  early_stopping_patience : int;
}

val create :
  ?alpha:float ->
  ?epsilon:float ->
  ?max_iter:int ->
  ?learning_rate:float ->
  ?batch_size:int ->
  ?num_iterations:int ->
  ?early_stopping_patience:int ->
  unit -> t

val to_json : t -> Yojson.Safe.t
val of_json : Yojson.Safe.t -> t