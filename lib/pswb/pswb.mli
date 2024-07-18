module type Config = sig
  val num_workers : int
  val num_points : int
  val step_size : float
  val max_iterations : int
end

module Make (C : Config) : sig
  val run : unit -> unit Lwt.t
end