open Torch
open Yojson.Safe.Util

type t =
  | Linear of { input_dim : int; output_dim : int }
  | MLP of { input_dim : int; hidden_dims : int list; output_dim : int }

let create = function
  | Linear { input_dim; output_dim } ->
      let vs = Var_store.create ~name:"linear_model" () in
      let linear = Layer.linear vs ~input_dim output_dim in
      (fun x -> Layer.forward linear x), vs
  | MLP { input_dim; hidden_dims; output_dim } ->
      let vs = Var_store.create ~name:"mlp_model" () in
      let rec build_layers dims =
        match dims with
        | [] -> Layer.linear vs ~input_dim output_dim
        | d :: rest ->
            let layer = Layer.linear vs ~input_dim d in
            let next_layers = build_layers rest in
            Layer.sequential [layer; Layer.relu (); next_layers]
      in
      let layers = build_layers hidden_dims in
      (fun x -> Layer.forward layers x), vs

let to_string = function
  | Linear { input_dim; output_dim } ->
      Printf.sprintf "Linear(in=%d, out=%d)" input_dim output_dim
  | MLP { input_dim; hidden_dims; output_dim } ->
      Printf.sprintf "MLP(in=%d, hidden=%s, out=%d)"
        input_dim
        (String.concat "," (List.map string_of_int hidden_dims))
        output_dim

let of_json json =
  match json |> member "model_type" |> to_string with
  | "Linear" ->
      Linear {
        input_dim = json |> member "input_dim" |> to_int;
        output_dim = json |> member "output_dim" |> to_int;
      }
  | "MLP" ->
      MLP {
        input_dim = json |> member "input_dim" |> to_int;
        hidden_dims = json |> member "hidden_dims" |> to_list |> List.map to_int;
        output_dim = json |> member "output_dim" |> to_int;
      }
  | _ -> invalid_arg "Unknown model type"