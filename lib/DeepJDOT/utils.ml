open Torch

let pad_to_size tensor target_size =
  let current_size = (Tensor.shape tensor).(2) in
  let pad_size = (target_size - current_size) / 2 in
  Tensor.pad tensor ~pad:[pad_size; pad_size; pad_size; pad_size] ~mode:"constant" ~value:0.0

let resize_tensor tensor new_size =
  Tensor.upsample_nearest2d tensor ~output_size:[new_size; new_size] ~scales_h:None ~scales_w:None