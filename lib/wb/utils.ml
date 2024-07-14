open Torch
open Types

let project_onto_simplex tensor =
  let sorted, _ = Tensor.sort tensor ~descending:true ~stable:true ~dim:0 in
  let cumsum = Tensor.cumsum sorted ~dim:0 ~dtype:(Tensor.kind tensor) in
  let arange = Tensor.arange ~start:1 ~end_:(Tensor.shape tensor).[0] ~options:(Tensor.device tensor, Tensor.kind tensor) in
  let gt = Tensor.(sorted + ((cumsum - arange) / arange) > 1.) in
  let k = Tensor.sum gt ~dim:0 ~keepdim:true in
  let tau = Tensor.((cumsum - 1.) / arange).(k - 1) in
  Tensor.max tensor (tau |> Tensor.expand_as tensor)

let project_onto_cube tensor =
  Tensor.clamp tensor ~min:(-1.) ~max:1.