open Torch

exception DimensionMismatch of string

let kl_div x y =
  if Tensor.(size x <> size y) then
    raise (DimensionMismatch "KL divergence inputs must have the same shape")
  else
    Tensor.(sum (x * log (x / y)))

let entropy x =
  Tensor.(mean (- x * log x))

let sinkhorn ?(num_iterations=100) ?(epsilon=1e-3) ?(tol=1e-9) cost mu_s mu_t =
  let n_s = Tensor.size mu_s 0 in
  let n_t = Tensor.size mu_t 0 in
  let k = Tensor.(exp (- cost / epsilon)) in
  let u = Tensor.ones [n_s] in
  let v = Tensor.ones [n_t] in
  
  let rec iterate i u v =
    if i = 0 then (u, v)
    else
      let u' = Tensor.(mu_s / (k * v)) in
      let v' = Tensor.(mu_t / (transpose k 0 1 * u')) in
      let err = Tensor.(max (abs (u' - u)) + max (abs (v' - v))) in
      if Tensor.to_float0_exn err < tol then (u', v')
      else iterate (i - 1) u' v'
  in
  
  let (u, v) = iterate num_iterations u v in
  Tensor.(k * u.unsqueeze 1 * v.unsqueeze 0)

let compute_cost x_s x_t =
  let n_s = Tensor.size x_s 0 in
  let n_t = Tensor.size x_t 0 in
  let x_s_exp = Tensor.expand x_s ~size:[n_s; n_t; -1] in
  let x_t_exp = Tensor.expand x_t ~size:[n_s; n_t; -1] in
  Tensor.(sum (pow (x_s_exp - x_t_exp) (Scalar.f 2.)) ~dim:[2])

let normalize_tensor t =
  let sum = Tensor.sum t in
  Tensor.(t / sum)

let check_probability_vector t =
  let sum = Tensor.sum t in
  let diff = Tensor.abs (Tensor.sub sum (Tensor.of_float 1.)) in
  Tensor.to_float0_exn diff < 1e-6