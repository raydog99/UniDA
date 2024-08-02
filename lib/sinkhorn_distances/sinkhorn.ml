open Torch
open Distance

type regularization =
  | Entropic
  | Quadratic

let epsilon = 1e-6

let is_close (a : Tensor.t) (b : Tensor.t) ~rtol ~atol =
  let diff = Tensor.abs (Tensor.sub a b) in
  let tol = Tensor.add (Tensor.mul_scalar b rtol) atol in
  Tensor.all (Tensor.le diff tol)

let ensure_non_negative (t : Tensor.t) =
  Tensor.max t (Tensor.full (Tensor.shape t) epsilon)

let apply_regularization (m : Tensor.t) (lambda : float) (reg_type : regularization) =
  match reg_type with
  | Entropic -> Tensor.exp (Tensor.mul_scalar m (Float.neg lambda))
  | Quadratic -> Tensor.div (Tensor.neg m) lambda

let sinkhorn_distance (m : Tensor.t) (lambda : float) (r : Tensor.t) (c : Tensor.t) (reg_type : regularization) ~max_iter ~tol =
  let batch_size, n, _ = Tensor.shape m in
  let k = apply_regularization m lambda reg_type in
  let u = Tensor.ones [batch_size; n; 1] in
  
  let rec iterate i u =
    if i >= max_iter then u
    else
      let v = Tensor.div c (Tensor.bmm (Tensor.transpose k ~dim0:1 ~dim1:2) u) in
      let u' = Tensor.div r (Tensor.bmm k v) in
      let u' = ensure_non_negative u' in
      if is_close u u' ~rtol:tol ~atol:epsilon then u'
      else iterate (i + 1) u'
  in
  
  let u_final = iterate 0 u in
  let v_final = Tensor.div c (Tensor.bmm (Tensor.transpose k ~dim0:1 ~dim1:2) u_final) in
  let p = Tensor.mul (Tensor.mul u_final (Tensor.transpose k ~dim0:1 ~dim1:2)) (Tensor.transpose v_final ~dim0:1 ~dim1:2) in
  Tensor.sum (Tensor.mul p m) ~dim:[1; 2]

let sinkhorn_divergence (m : Tensor.t) (lambda : float) (r : Tensor.t) (c : Tensor.t) (reg_type : regularization) ~max_iter ~tol =
  let d_rc = sinkhorn_distance m lambda r c reg_type ~max_iter ~tol in
  let d_rr = sinkhorn_distance m lambda r r reg_type ~max_iter ~tol in
  let d_cc = sinkhorn_distance m lambda c c reg_type ~max_iter ~tol in
  Tensor.sub (Tensor.mul_scalar d_rc 2.0) (Tensor.add d_rr d_cc)

let ot_distance cost_fn lambda x y reg_type ~max_iter ~tol =
  let m = cost_fn x y in
  let batch_size, n, _ = Tensor.shape m in
  let r = Tensor.full [batch_size; n; 1] (1. /. Float.of_int n) in
  let c = Tensor.full [batch_size; n; 1] (1. /. Float.of_int n) in
  sinkhorn_divergence m lambda r c reg_type ~max_iter ~tol

let sinkhorn_gradient cost_fn lambda x y reg_type ~max_iter ~tol =
  let x = Tensor.set_requires_grad x true in
  let y = Tensor.set_requires_grad y true in
  let loss = ot_distance cost_fn lambda x y reg_type ~max_iter ~tol in
  let grad_x, grad_y = Tensor.grad loss [x; y] in
  (loss, grad_x, grad_y)

let normalize (t : Tensor.t) ~dim =
  let sum = Tensor.sum t ~dim:[dim] ~keepdim:true in
  Tensor.div t sum

let create_histogram (data : Tensor.t) (bins : int) =
  let min_val = Tensor.min data in
  let max_val = Tensor.max data in
  let step = Tensor.div (Tensor.sub max_val min_val) (Tensor.scalar_float (Float.of_int bins)) in
  let hist = Tensor.histogram data bins ~min:min_val ~max:max_val in
  normalize hist ~dim:0