type t = {
  epsilon1: float;
  epsilon2: float;
  mu: float;
  max_iter: int;
  tolerance: float;
  num_workers: int;
  log_level: Logs.level;
}

let create ?(epsilon1=0.1) ?(epsilon2=0.1) ?(mu=1.0) ?(max_iter=100) ?(tolerance=1e-6) ?(num_workers=4) ?(log_level=Logs.Info) () =
  { epsilon1; epsilon2; mu; max_iter; tolerance; num_workers; log_level }

let get_epsilon1 config = config.epsilon1
let get_epsilon2 config = config.epsilon2
let get_mu config = config.mu
let get_max_iter config = config.max_iter
let get_tolerance config = config.tolerance
let get_num_workers config = config.num_workers
let get_log_level config = config.log_level