open Torch

type t = {
  x: Tensor.t;
  y: Tensor.t;
  z: Tensor.t;
  grad_x: Tensor.t;
  grad_y: Tensor.t;
  A: Tensor.t;
  b: Tensor.t;
  c: Tensor.t;
  eta_x: float;
  eta_y: float;
  gamma: float;
}

let create x y A b c eta_x eta_y gamma =
  let grad_x = Tensor.zeros_like x in
  let grad_y = Tensor.zeros_like y in
  let z = Tensor.clone x in
  { x; y; z; grad_x; grad_y; A; b; c; eta_x; eta_y; gamma }

let update_gradients t =
  let grad_x = Tensor.add (Tensor.matmul t.A t.y) t.c in
  let grad_y = Tensor.sub (Tensor.matmul (Tensor.transpose t.A ~dim0:0 ~dim1:1) t.x) t.b in
  { t with grad_x; grad_y }

let update_primal t =
  let x_new = Tensor.sub t.x (Tensor.mul_scalar t.grad_x (Scalar.float t.eta_x)) in
  let z_new = Tensor.add x_new (Tensor.mul_scalar (Tensor.sub x_new t.x) (Scalar.float t.gamma)) in
  { t with x = x_new; z = z_new }

let update_dual t =
  let y_new = Tensor.add t.y (Tensor.mul_scalar t.grad_y (Scalar.float t.eta_y)) in
  { t with y = y_new }

let run t max_iter =
  let rec loop iter t =
    if iter >= max_iter then t
    else
      let t = update_gradients t in
      let t = update_primal t in
      let t = update_dual t in
      loop (iter + 1) t
  in
  loop 0 t

let get_primal_result t = t.x
let get_dual_result t = t.y