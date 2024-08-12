open Torch
open Base

type layer = {
  linear: Layer.t;
  activation: Tensor.t -> Tensor.t;
}

type t = {
  layers: layer list;
  output_linear: Layer.t;
}

let create input_dim hidden_dims output_dim =
  let rec build_layers in_dim dims acc =
    match dims with
    | [] -> List.rev acc
    | out_dim :: rest ->
        let linear = Layer.linear in_dim out_dim ~bias:false in
        let layer = { linear; activation = Tensor.relu_ } in
        build_layers out_dim rest (layer :: acc)
  in
  let layers = build_layers input_dim hidden_dims [] in
  let output_linear = Layer.linear (List.last_exn hidden_dims) output_dim ~bias:false in
  { layers; output_linear }

let forward icnn x =
  let rec forward_layers x = function
    | [] -> x
    | layer :: rest ->
        let z = Layer.forward layer.linear x in
        let a = layer.activation z in
        forward_layers (Tensor.(x + a)) rest
  in
  let y = forward_layers x icnn.layers in
  Layer.forward icnn.output_linear y

let gradient icnn x =
  let y = forward icnn x in
  Tensor.grad y [| x |].(0)

let parameters icnn =
  let layer_params = List.concat_map icnn.layers ~f:(fun layer -> Layer.parameters layer.linear) in
  layer_params @ Layer.parameters icnn.output_linear

let input_dim icnn =
  let first_layer = List.hd_exn icnn.layers in
  Layer.input_dim first_layer.linear

let output_dim icnn =
  Layer.output_dim icnn.output_linear

let state_dict icnn =
  let layer_states = List.mapi icnn.layers ~f:(fun i layer ->
    (Printf.sprintf "layer_%d" i, Layer.state_dict layer.linear)
  ) in
  ("output_linear", Layer.state_dict icnn.output_linear) :: layer_states

let load state_dict =
  let layers = List.filter_map state_dict ~f:(fun (name, state) ->
    if String.is_prefix name ~prefix:"layer_" then
      Some { linear = Layer.load_state_dict state; activation = Tensor.relu_ }
    else
      None
  ) in
  let output_linear = Layer.load_state_dict (List.Assoc.find_exn state_dict ~equal:String.equal "output_linear") in
  { layers; output_linear }