open Torch

let normalize_tensor (t : Tensor.t) : Tensor.t =
  Tensor.(t / sum t)

let is_probability_vector (t : Tensor.t) : bool =
  let sum = Tensor.(sum t |> to_float0_exn) in
  abs_float (sum -. 1.0) < 1e-6

let generate_random_marginal (size : int) : Tensor.t =
  let t = Tensor.rand [size] in
  normalize_tensor t

let generate_random_cost_matrix (m : int) (n : int) : Tensor.t =
  Tensor.rand [m; n]

let tensor_to_list (t : Tensor.t) : float list =
  Tensor.to_float1_exn t |> Array.to_list

let list_to_tensor (l : float list) : Tensor.t =
  Tensor.of_float1 (Array.of_list l)

let print_tensor (t : Tensor.t) : unit =
  Tensor.print t