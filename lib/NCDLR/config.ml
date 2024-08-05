open Types
open Yojson.Basic.Util

let default_config = {
  input_dim = 3 * 32 * 32;
  output_dim = 128;
  num_epochs = 100;
  batch_size = 256;
  learning_rate = 0.001;
  gamma = 0.1;
  beta = 0.5;
  max_k_u = 100;
  buffer_size = 2048;
}

let load_config filename =
  try
    let json = Yojson.Basic.from_file filename in
    {
      input_dim = json |> member "input_dim" |> to_int;
      output_dim = json |> member "output_dim" |> to_int;
      num_epochs = json |> member "num_epochs" |> to_int;
      batch_size = json |> member "batch_size" |> to_int;
      learning_rate = json |> member "learning_rate" |> to_float;
      gamma = json |> member "gamma" |> to_float;
      beta = json |> member "beta" |> to_float;
      max_k_u = json |> member "max_k_u" |> to_int;
      buffer_size = json |> member "buffer_size" |> to_int;
    }
  with
  | Sys_error _ -> Printf.printf "Config file not found. Using default config.\n"; default_config
  | Json_error _ -> Printf.printf "Invalid JSON in config file. Using default config.\n"; default_config
  | _ -> Printf.printf "Error reading config file. Using default config.\n"; default_config