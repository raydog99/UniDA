open Torch
open Utils
open Model
open Dataset

let create_long_tailed_ratio num_classes imbalance_factor =
  let ratios = Array.make num_classes 1.0 in
  for i = 1 to num_classes - 1 do
    ratios.(i) <- ratios.(i-1) /. (imbalance_factor ** (1.0 /. float_of_int (num_classes - 1)))
  done;
  Tensor.of_float1 ratios

let create_teacher_based_ratio model dataset =
  let features, _ = Dataset.get_all dataset in
  let logits = Model.forward model features in
  Utils.softmax logits 1