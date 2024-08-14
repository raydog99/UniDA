open Torch

type t = {
  fc1: Layer.t;
  fc2: Layer.t;
}

let create input_dim num_classes =
  try
    let fc1 = Layer.linear ~in_features:input_dim ~out_features:128 () in
    let fc2 = Layer.linear ~in_features:128 ~out_features:num_classes () in
    Ok { fc1; fc2 }
  with
  | exn -> Error (Printf.sprintf "Failed to create classifier: %s" (Printexc.to_string exn))

let forward t x =
  try
    Ok (
      let open Layer in
      x
      |> linear t.fc1
      |> relu
      |> linear t.fc2
      |> softmax ~dim:(-1)
    )
  with
  | exn -> Error (Printf.sprintf "Forward pass failed in classifier: %s" (Printexc.to_string exn))