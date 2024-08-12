open Torch
open Base

type dataset = {
  x: Tensor.t;
  y: Tensor.t;
}

let create_dataset ~num_samples ~dim ~noise_scale =
  let x = Tensor.randn [num_samples; dim] in
  let y = Tensor.(x + (randn [num_samples; dim] * f noise_scale)) in
  { x; y }

let data_loader dataset ~batch_size ~shuffle =
  Dataset.of_tensors ~device:Cuda [dataset.x; dataset.y]
  |> Dataset.map ~f:(fun [x; y] -> (x, y))
  |> (if shuffle then Dataset.shuffle else Fn.id)
  |> Dataset.batch ~batch_size

let split_dataset dataset ~train_ratio =
  let num_samples = (Tensor.shape dataset.x).(0) in
  let num_train = Float.to_int (Float.of_int num_samples *. train_ratio) in
  let train_x, val_x = Tensor.split2 ~dim:0 ~split_size:num_train dataset.x in
  let train_y, val_y = Tensor.split2 ~dim:0 ~split_size:num_train dataset.y in
  { x = train_x; y = train_y }, { x = val_x; y = val_y }