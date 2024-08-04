open Torch
open Csv

type t = {
  xs: Tensor.t;
  ys: Tensor.t;
  xt: Tensor.t;
  yt: Tensor.t option;
}

let load_data ~source_file ~target_file =
  let load_csv file =
    let data = Csv.load file in
    let tensor = Tensor.of_float2 (Array.of_list (List.map Array.of_list data)) in
    let features = Tensor.narrow tensor ~dim:1 ~start:0 ~length:(Tensor.shape2_exn tensor |> snd |> pred) in
    let labels = Tensor.narrow tensor ~dim:1 ~start:(Tensor.shape2_exn tensor |> snd |> pred) ~length:1 in
    (features, labels)
  in
  let xs, ys = load_csv source_file in
  let xt, yt = load_csv target_file in
  { xs; ys; xt; yt = Some yt }

let normalize_features tensor =
  let mean = Tensor.mean tensor ~dim:[0] ~keepdim:true in
  let std = Tensor.std tensor ~dim:[0] ~keepdim:true in
  Tensor.((sub tensor mean) / std)

let preprocess data =
  let xs_norm = normalize_features data.xs in
  let xt_norm = normalize_features data.xt in
  { data with xs = xs_norm; xt = xt_norm }

let create_batches data batch_size =
  let num_samples = Tensor.shape data.xs |> List.hd in
  let num_batches = num_samples / batch_size in
  List.init num_batches (fun i ->
    let start_idx = i * batch_size in
    let end_idx = min (start_idx + batch_size) num_samples in
    let length = end_idx - start_idx in
    let xs_batch = Tensor.narrow data.xs ~dim:0 ~start:start_idx ~length in
    let ys_batch = Tensor.narrow data.ys ~dim:0 ~start:start_idx ~length in
    let xt_batch = Tensor.narrow data.xt ~dim:0 ~start:start_idx ~length in
    (xs_batch, ys_batch, xt_batch)
  )