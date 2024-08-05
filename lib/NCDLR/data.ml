open Torch
open Types

let load_dataset path : dataset =
  try
    let npz = Npz.open_in path in
    let images = Npz.read_tensor npz "images" in
    let labels = Npz.read_tensor npz "labels" in
    Npz.close_in npz;
    { images; labels }
  with
  | Sys_error msg -> failwith ("Error loading dataset: " ^ msg)
  | _ -> failwith "Unknown error while loading dataset"

let sample_batch (dataset : dataset) batch_size =
  let num_samples = Tensor.shape dataset.images |> List.hd in
  let indices = Tensor.randint ~high:num_samples [batch_size] in
  let images = Tensor.index_select dataset.images ~dim:0 ~index:indices in
  let labels = Tensor.index_select dataset.labels ~dim:0 ~index:indices in
  images, labels