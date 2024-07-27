open Torch

let check_input_dimensions source_features target_features =
  let source_shape = Tensor.shape source_features in
  let target_shape = Tensor.shape target_features in
  if List.length source_shape <> 2 || List.length target_shape <> 2 then
    failwith "Input tensors must be 2-dimensional"
  else if List.nth source_shape 1 <> List.nth target_shape 1 then
    failwith "Source and target features must have the same number of dimensions"

let normalize_features features =
  let mean = Tensor.mean features ~dim:[0] ~keepdim:true in
  let std = Tensor.std features ~dim:[0] ~keepdim:true ~unbiased:false in
  Tensor.div (Tensor.sub features mean) (Tensor.add std (Tensor.f 1e-8))

let create_random_features num_samples num_features =
  Tensor.randn [num_samples; num_features]