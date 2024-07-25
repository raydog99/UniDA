open Torch

let sinkhorn (cost : Tensor.t) (a : Tensor.t) (b : Tensor.t) (epsilon : float) (max_iter : int) : Tensor.t =
  let m, n = Tensor.shape2_exn cost in
  let k = Tensor.exp Tensor.(neg cost / epsilon) in
  let u = Tensor.ones [m; 1] in
  let v = Tensor.ones [n; 1] in
  
  let rec loop i =
    if i >= max_iter then (u, v)
    else
      let u_new = Tensor.(a / (matmul k v)) in
      let v_new = Tensor.(b / (matmul (transpose k ~dim0:0 ~dim1:1) u_new)) in
      loop (i + 1)
  in
  
  let u, v = loop 0 in
  Tensor.(u * k * transpose v ~dim0:0 ~dim1:1)

let entropy_regularized_ot (x : Tensor.t) (y : Tensor.t) (cost_fn : Tensor.t -> Tensor.t -> Tensor.t) (epsilon : float) (max_iter : int) : Tensor.t =
  let cost = cost_fn x y in
  let a = Tensor.ones [Tensor.size x ~dim:0; 1] in
  let b = Tensor.ones [Tensor.size y ~dim:0; 1] in
  sinkhorn cost a b epsilon max_iter