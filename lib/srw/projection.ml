open Torch

let project_onto_R (tensor : Tensor.t) (k : int) : Tensor.t =
  let d = Tensor.size tensor ~dim:0 in
  let eigenvalues, eigenvectors = Tensor.symeig tensor ~eigenvectors:true in
  let sorted_indices = Tensor.argsort eigenvalues ~descending:true in
  let top_k_indices = Tensor.narrow sorted_indices ~dim:0 ~start:0 ~length:k in
  let top_k_eigenvectors = Tensor.index_select eigenvectors ~dim:1 ~index:top_k_indices in
  Tensor.matmul top_k_eigenvectors (Tensor.transpose top_k_eigenvectors ~dim0:1 ~dim1:0)

let dykstra_projection (tensor : Tensor.t) (k : int) (max_iter : int) : Tensor.t =
  let d = Tensor.size tensor ~dim:0 in
  let identity = Tensor.eye d in
  let rec loop i prev_p =
    if i >= max_iter then prev_p
    else
      let y = Tensor.(max (min tensor identity) (Tensor.zeros [d; d])) in
      let p = project_onto_R y k in
      let next_tensor = Tensor.(tensor + p - y) in
      loop (i + 1) p
  in
  loop 0 (Tensor.zeros [d; d])