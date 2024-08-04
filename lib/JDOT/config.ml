open Yojson.Safe.Util

type t = {
  alpha : float;
  epsilon : float;
  max_iter : int;
  learning_rate : float;
  batch_size : int;
  num_iterations : int;
  early_stopping_patience : int;
}

let create ?(alpha=1.0) ?(epsilon=0.1) ?(max_iter=100) 
           ?(learning_rate=0.001) ?(batch_size=32) 
           ?(num_iterations=1000) ?(early_stopping_patience=10) () =
  { alpha; epsilon; max_iter; learning_rate; batch_size; num_iterations; early_stopping_patience }

let to_json config =
  `Assoc [
    "alpha", `Float config.alpha;
    "epsilon", `Float config.epsilon;
    "max_iter", `Int config.max_iter;
    "learning_rate", `Float config.learning_rate;
    "batch_size", `Int config.batch_size;
    "num_iterations", `Int config.num_iterations;
    "early_stopping_patience", `Int config.early_stopping_patience;
  ]

let of_json json =
  {
    alpha = json |> member "alpha" |> to_float;
    epsilon = json |> member "epsilon" |> to_float;
    max_iter = json |> member "max_iter" |> to_int;
    learning_rate = json |> member "learning_rate" |> to_float;
    batch_size = json |> member "batch_size" |> to_int;
    num_iterations = json |> member "num_iterations" |> to_int;
    early_stopping_patience = json |> member "early_stopping_patience" |> to_int;
  }