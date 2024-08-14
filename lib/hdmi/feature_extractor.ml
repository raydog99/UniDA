(* File: lib/feature_extractor.ml *)

open Torch

type t = {
  conv1: Layer.t;
  conv2: Layer.t;
  fc: Layer.t;
}

let create input_channels =
  try
    let conv1 = Layer.conv2d ~in_channels:input_channels ~out_channels:64 ~kernel_size:3 ~stride:1 ~padding:1 () in
    let conv2 = Layer.conv2d ~in_channels:64 ~out_channels:128 ~kernel_size:3 ~stride:1 ~padding:1 () in
    let fc = Layer.linear ~in_features:128 ~out_features:256 () in
    Ok { conv1; conv2; fc }
  with
  | exn -> Error (Printf.sprintf "Failed to create feature extractor: %s" (Printexc.to_string exn))

let forward t x =
  try
    Ok (
      let open Layer in
      x
      |> conv2d t.conv1
      |> relu
      |> max_pool2d ~kernel_size:2 ~stride:2
      |> conv2d t.conv2
      |> relu
      |> max_pool2d ~kernel_size:2 ~stride:2
      |> flatten
      |> linear t.fc
      |> relu
    )
  with
  | exn -> Error (Printf.sprintf "Forward pass failed in feature extractor: %s" (Printexc.to_string exn))