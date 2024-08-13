open Torch

let sample_empirical_measure measure sample_size =
  let n = Tensor.shape measure |> List.hd in
  let indices = Tensor.randint ~high:n [sample_size] in
  Tensor.index_select measure ~dim:0 ~index:indices

let create_measure_from_points points =
  let n = Tensor.shape points |> List.hd in
  Tensor.full [n] (1. /. float_of_int n)

let normalize_measure measure =
  let sum = Tensor.sum measure in
  Tensor.div measure sum

let histogram_to_measure histogram =
  let sum = Tensor.sum histogram in
  Tensor.div histogram sum