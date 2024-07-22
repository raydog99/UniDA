open Torch

type cost_matrix = Tensor.t
type transport_plan = Tensor.t
type marginals = Tensor.t * Tensor.t

(** Compute the delta_omega function for the given regularization *)
let delta_omega reg x =
  Regularization.validate reg;
  match reg with
  | Regularization.NegativeEntropy gamma ->
      Tensor.mul_scalar (Tensor.sum (Tensor.exp (Tensor.div_scalar x gamma))) gamma
  | Regularization.SquaredNorm gamma ->
      let x_pos = Tensor.relu x in
      Tensor.div_scalar (Tensor.sum (Tensor.pow x_pos 2.)) (2. *. gamma)
  | Regularization.GroupLasso (gamma, mu, groups) ->
      let x_pos = Tensor.relu x in
      let group_term = 
        List.fold_left (fun acc group ->
          let x_group = Tensor.index_select x_pos 0 (Tensor.of_int1 group) in
          Tensor.add acc (Tensor.norm x_group)
        ) (Tensor.zeros []) groups
      in
      Tensor.add 
        (Tensor.div_scalar (Tensor.sum (Tensor.pow x_pos 2.)) (2. *. gamma))
        (Tensor.mul_scalar group_term mu)

(** Compute the gradient of delta_omega for the given regularization *)
let grad_delta_omega reg x =
  Regularization.validate reg;
  match reg with
  | Regularization.NegativeEntropy gamma ->
      Tensor.div_scalar (Tensor.exp (Tensor.div_scalar x gamma)) gamma
  | Regularization.SquaredNorm gamma ->
      Tensor.div_scalar (Tensor.relu x) gamma
  | Regularization.GroupLasso (gamma, mu, groups) ->
      let x_pos = Tensor.relu x in
      let group_grad = 
        List.fold_left (fun acc group ->
          let x_group = Tensor.index_select x_pos 0 (Tensor.of_int1 group) in
          let norm = Tensor.norm x_group in
          let grad_group = Tensor.div x_group (Tensor.add norm 1e-10) in
          Tensor.index_add acc 0 (Tensor.of_int1 group) grad_group
        ) (Tensor.zeros_like x) groups
      in
      Tensor.add 
        (Tensor.div_scalar x_pos gamma)
        (Tensor.mul_scalar group_grad mu)

(** Compute the max_omega function for the given regularization *)
let max_omega reg x =
  Regularization.validate reg;
  match reg with
  | Regularization.NegativeEntropy gamma ->
      let max_x = Tensor.max x in
      let exp_terms = Tensor.exp (Tensor.sub x max_x) in
      Tensor.add (Tensor.mul_scalar (Tensor.log (Tensor.sum exp_terms)) gamma) max_x
  | Regularization.SquaredNorm gamma ->
      let proj = Tensor.div_scalar x gamma in
      let proj_simplex = Tensor.softmax proj ~dim:0 in
      Tensor.dot x proj_simplex
  | Regularization.GroupLasso (gamma, mu, groups) ->
      (* Approximate solution using proximal gradient method *)
      let n = Tensor.shape x |> List.hd in
      let y = Tensor.ones [n] |> Tensor.div_scalar (float n) in
      let max_iter = 100 in
      let step_size = 1. /. gamma in
      let rec iterate y iter =
        if iter >= max_iter then y
        else
          let grad = Tensor.sub (Tensor.div_scalar y gamma) x in
          let y_new = Tensor.sub y (Tensor.mul_scalar grad step_size) in
          let y_proj = project_group_lasso y_new mu groups in
          iterate y_proj (iter + 1)
      in
      let result = iterate y 0 in
      Tensor.dot x result

(** Project onto the group lasso constraint *)
and project_group_lasso y mu groups =
  List.fold_left (fun acc group ->
    let y_group = Tensor.index_select y 0 (Tensor.of_int1 group) in
    let norm = Tensor.norm y_group in
    let scale = Tensor.max (Tensor.ones []) (Tensor.div norm mu) in
    let y_proj = Tensor.div y_group scale in
    Tensor.index_copy acc 0 (Tensor.of_int1 group) y_proj
  ) y groups

(** Compute the gradient of max_omega for the given regularization *)
let grad_max_omega reg x =
  Regularization.validate reg;
  match reg with
  | Regularization.NegativeEntropy gamma ->
      let max_x = Tensor.max x in
      let exp_terms = Tensor.exp (Tensor.sub x max_x) in
      Tensor.div exp_terms (Tensor.sum exp_terms)
  | Regularization.SquaredNorm gamma ->
      Tensor.softmax (Tensor.div_scalar x gamma) ~dim:0
  | Regularization.GroupLasso (gamma, mu, groups) ->
      let y = max_omega reg x in
      Tensor.div y (Tensor.sum y)

(** Optimize using L-BFGS algorithm *)
let optimize_lbfgs objective grad init_params max_iter =
  let rec iterate params iter =
    if iter >= max_iter then params
    else
      let loss = objective params in
      let g = grad params in
      let params_new = Torch_optim.Lbfgs.step objective grad params in
      iterate params_new (iter + 1)
  in
  iterate init_params 0

(** Compute the optimal transport plan with regularization *)
let ot_omega a b c reg =
  let m, n = Tensor.shape c in
  if Tensor.shape a <> [m] || Tensor.shape b <> [n] then
    invalid_arg "Incompatible dimensions for marginals and cost matrix";
  let init_plan = Tensor.ones [m; n] |> Tensor.div_scalar (float (m * n)) in
  let objective t =
    let transport_cost = Tensor.sum (Tensor.mul t c) in
    let regularization = Regularization.apply reg t in
    Tensor.add transport_cost regularization
  in
  let grad t =
    let grad_cost = c in
    let grad_reg = Tensor.grad (Regularization.apply reg) t in
    Tensor.add grad_cost grad_reg
  in
  optimize_lbfgs objective grad init_plan 100

(** Compute the smoothed semi-dual solution *)
let smoothed_semi_dual a b c reg =
  let m, n = Tensor.shape c in
  if Tensor.shape a <> [m] || Tensor.shape b <> [n] then
    invalid_arg "Incompatible dimensions for marginals and cost matrix";
  let alpha = Tensor.zeros [m] in
  let objective alpha =
    let term1 = Tensor.dot a alpha in
    let term2 = 
      Tensor.sum (Tensor.map2 (fun b_j c_j ->
        let x = Tensor.sub alpha c_j in
        Tensor.mul_scalar (max_omega reg x) b_j
      ) b (Tensor.unbind c ~dim:1))
    in
    Tensor.sub term1 term2
  in
  let grad alpha =
    let grad_term1 = a in
    let grad_term2 = 
      Tensor.sum (Tensor.map2 (fun b_j c_j ->
        let x = Tensor.sub alpha c_j in
        Tensor.mul_scalar (grad_max_omega reg x) b_j
      ) b (Tensor.unbind c ~dim:1))
    in
    Tensor.sub grad_term1 grad_term2
  in
  optimize_lbfgs objective grad alpha 100

(** Compute Wasserstein distance *)
let wasserstein_distance t c =
  Tensor.sum (Tensor.mul t c)

(** Generate random cost matrix *)
let random_cost_matrix m n =
  Tensor.rand [m; n]

(** Generate random marginals *)
let random_marginals m n =
  let a = Tensor.rand [m] in
  let b = Tensor.rand [n] in
  (Tensor.div a (Tensor.sum a), Tensor.div b (Tensor.sum b))