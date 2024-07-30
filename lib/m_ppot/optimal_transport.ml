open Torch

let compute_cost_matrix (prototypes : Tensor.t) (target_features : Tensor.t) : Tensor.t =
  let expanded_prototypes = Tensor.unsqueeze prototypes ~dim:1 in
  let expanded_target_features = Tensor.unsqueeze target_features ~dim:0 in
  Tensor.mse_loss expanded_prototypes expanded_target_features Reduction.None

let sinkhorn (cost_matrix : Tensor.t) (alpha : float) (num_iterations : int) (epsilon : float) : Tensor.t =
  let n, m = Tensor.shape2_exn cost_matrix in
  let mu = Tensor.full [n] (1. /. float_of_int n) ~device:(Tensor.device cost_matrix) in
  let nu = Tensor.full [m] (alpha /. float_of_int m) ~device:(Tensor.device cost_matrix) in

  let rec iterate u v k =
    if k = 0 then (u, v)
    else
      let u' = mu /. Tensor.(sum (v * ((-cost_matrix /. epsilon) |> exp)) ~dim:[1]) in
      let v' = nu /. Tensor.(sum (u' * ((-cost_matrix /. epsilon) |> exp)) ~dim:[0]) in
      iterate u' v' (k - 1)
  in

  let u, v = iterate (Tensor.ones [n] ~device:(Tensor.device cost_matrix)) 
                     (Tensor.ones [m] ~device:(Tensor.device cost_matrix)) 
                     num_iterations in

  Tensor.(u.unsqueeze 1 * ((-cost_matrix /. epsilon) |> exp) * v)

let solve_pot (cost_matrix : Tensor.t) (alpha : float) : Tensor.t =
  sinkhorn cost_matrix alpha 100 0.01