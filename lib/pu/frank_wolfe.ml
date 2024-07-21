open Torch

type optimization_problem = Tensor.t -> float * Tensor.t

let line_search f x d =
  let rec search alpha =
    let alpha' = alpha /. 2. in
    if alpha' < 1e-16 then alpha
    else
      let fx, _ = f x in
      let fx', _ = f (Tensor.add x (Tensor.mul d alpha')) in
      if fx' < fx then alpha' else search alpha'
  in
  search 1.

let optimize f x0 max_iter tol callback =
  let rec iterate x k =
    let fx, grad = f x in
    (match callback with Some cb -> cb k x | None -> ());
    if k >= max_iter then x
    else
      let d = Tensor.neg grad in
      let alpha = line_search f x d in
      let x_new = Tensor.add x (Tensor.mul d alpha) in
      if Tensor.norm (Tensor.sub x_new x) ~p:2 |> Tensor.item < tol then x_new
      else iterate x_new (k + 1)
  in
  iterate x0 0

let partial_gw_step cs ct p q s x =
  let loss = PartialOT.compute_partial_gw_loss cs ct x in
  let grad = PartialOT.gradient_partial_gw cs ct x in
  let linear_problem = Utils.solve_linear_program grad p q in
  loss, Tensor.sub linear_problem x