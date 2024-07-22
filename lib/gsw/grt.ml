open Torch

let transform samples defining_func omega_theta =
  let batch_size = Tensor.shape samples |> List.hd in
  let num_projections = Tensor.shape omega_theta |> List.hd in
  
  let grt = Tensor.zeros [batch_size; num_projections] in
  
  for i = 0 to num_projections - 1 do
    let theta = Tensor.select omega_theta 0 i in
    let projected = defining_func samples theta in
    Tensor.copy_ (Tensor.select grt 1 i) projected
  done;
  
  grt