open Torch

module Madaot = struct
  type t = {
    delta : float;
    zeta : float;
    thd1 : float;
    thd2 : float;
    nIter : int;
    mutable coefs : Tensor.t option;
    mutable log : (string * Tensor.t) list option;
    mutable transport : Tensor.t option;
  }

  let create ?(delta=1.0) ?(zeta=1e-5) ?(thd1=1e-4) ?(thd2=1e-7) ?(nIter=10) () =
    { delta; zeta; thd1; thd2; nIter; coefs = None; log = None; transport = None }

  let fit t x_s y_s x_t =
    let coefs, log, transport = Learn_classif_adapt_grad.optimize x_s y_s x_t t.delta t.zeta t.coefs None t.thd1 t.thd2 t.nIter in
    t.coefs <- Some coefs;
    t.log <- Some log;
    t.transport <- Some transport;
    t

  let decision_func t x =
    match t.coefs with
    | Some coefs -> Tensor.matmul x coefs
    | None -> failwith "Model not fitted"

  let predict t x =
    Tensor.sign (decision_func t x)
end

module Obj_fun_smooth = struct
  let calculate w x_s y_s x_t zeta delta gammas inds class_weights =
    let src, grad_src = Source_hinge_smooth.calculate w x_s y_s class_weights in
    let disc_vec, grad_disc_vec = Align_corresp_abs_smooth.calculate w (Tensor.transpose x_s ~dim0:0 ~dim1:1) (Tensor.transpose x_t ~dim0:0 ~dim1:1) gammas inds in
    let reg = Tensor.dot w w in
    Tensor.(src + (scalar delta * disc_vec) + (scalar zeta * reg)),
    Tensor.(grad_src + (scalar delta * grad_disc_vec) + (scalar (2. *. zeta) * w))
end

module Learn_classif_adapt_grad = struct
  let optimize x_s y_s x_t delta zeta w_sol gamma_sol thd1 thd2 nIter =
    let class_weights, _ = Class_weights.calculate y_s in
    let d = Tensor.size x_s |> List.nth 1 in
    let w_sol = match w_sol with
      | Some w -> w
      | None -> Tensor.ones [1; d] |> Tensor.div_scalar (float_of_int d)
    in
    let n = Tensor.size x_t |> List.hd in
    
    let gamma_sol = match gamma_sol with
      | Some g -> g
      | None ->
          let dist_mat = Cdist.calculate x_s x_t `Sqeuclidean in
          Emd.calculate class_weights (Tensor.ones [1; n] |> Tensor.div_scalar (float_of_int n)) dist_mat
    in
    
    let gammas, inds = Transport_content.calculate gamma_sol in
    
    let rec optimize iter w_old gamma_old value_old =
      if iter >= nIter then w_old, [], gamma_old
      else
        let obj_fun = Obj_fun_smooth.calculate w_old x_s y_s x_t zeta delta gammas inds class_weights in
        let w_sol = Lbfgs.minimize obj_fun w_old in
        
        let new_value, _ = obj_fun w_sol in
        let err = (value_old -. Tensor.float_value new_value) /. max value_old (Tensor.float_value new_value) 1. in
        
        if abs_float err <= thd1 || Tensor.cosine_similarity w_sol w_old ~dim:0 ~eps:1e-8 < Tensor.of_float thd2 then
          w_sol, [], gamma_old
        else
          let gamma_sol, gammas, inds, _ = Minimax_ot_term_smooth.calculate w_sol x_s x_t class_weights gamma_old in
          optimize (iter + 1) w_sol gamma_sol (Tensor.float_value new_value)
    in
    
    optimize 0 w_sol gamma_sol (fst (Obj_fun_smooth.calculate w_sol x_s y_s x_t zeta delta gammas inds class_weights) |> Tensor.float_value)
end

module Source_hinge_smooth = struct
  let calculate w x_s y_s class_weights =
    let yxa = Tensor.(y_s * (matmul x_s w)) in
    let val_ = Tensor.(dot class_weights (softplus (scalar 1. - yxa))) in
    let signed_margin_viol = Tensor.(neg y_s * (sigmoid (scalar 1. - yxa))) in
    let grad = Tensor.(matmul (transpose class_weights ~dim0:0 ~dim1:1) (signed_margin_viol * x_s)) in
    val_, grad
end

module Align_corresp_abs_smooth = struct
  let calculate w x_s_t x_t_t gammas inds =
    let dxxa = Tensor.(x_s_t * (matmul w (transpose x_s_t ~dim0:0 ~dim1:1)) - x_t_t * (matmul w (transpose x_t_t ~dim0:0 ~dim1:1))) in
    let mean_abs_dxxa = Tensor.sum (Tensor.abs dxxa * gammas) ~dim:[1] in
    let soft_max_ind = Tensor.softmax (Tensor.mul_scalar mean_abs_dxxa 1e2) ~dim:0 in
    let sign_dxxa = Tensor.sign dxxa in
    let jac_per_pair = Tensor.(x_s_t * (matmul soft_max_ind (x_s_t * sign_dxxa)) - x_t_t * (matmul soft_max_ind (x_t_t * sign_dxxa))) in
    Tensor.logsumexp (Tensor.mul_scalar mean_abs_dxxa 1e2) ~dim:[0],
    Tensor.sum (jac_per_pair * gammas) ~dim:[1]
end

module Class_weights = struct
  let calculate y =
    let classes = Tensor.unique y in
    let binary_classes = Tensor.eq y classes in
    let class_counts = Tensor.sum binary_classes ~dim:[1] in
    let weights = Tensor.div (Tensor.sum (Tensor.div binary_classes class_counts) ~dim:[0]) (Tensor.of_float (float_of_int (Tensor.size classes |> List.hd))) in
    Tensor.div weights (Tensor.sum weights),
    Tensor.sub (Tensor.mul_scalar binary_classes 2.) (Tensor.of_float 1.)
end