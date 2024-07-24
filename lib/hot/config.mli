type t

val create :
  ?epsilon1:float ->
  ?epsilon2:float ->
  ?mu:float ->
  ?max_iter:int ->
  ?tolerance:float ->
  ?num_workers:int ->
  ?log_level:Logs.level ->
  unit -> t

val get_epsilon1 : t -> float
val get_epsilon2 : t -> float
val get_mu : t -> float
val get_max_iter : t -> int
val get_tolerance : t -> float
val get_num_workers : t -> int
val get_log_level : t -> Logs.level