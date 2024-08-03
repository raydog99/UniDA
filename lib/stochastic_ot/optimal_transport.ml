open Base
open Torch

module type Cost_fn = sig
  val cost : Tensor.t -> Tensor.t -> Tensor.t
end

module OT (C : Cost_fn) = struct
  let kl_divergence pi xi =
    Tensor.sum' (Tensor.mul pi (Tensor.log (Tensor.div pi xi))) |> Tensor.item

  let entropy pi =
    -1. *. (Tensor.sum' (Tensor.mul pi (Tensor.log pi)) |> Tensor.item)

  let primal_problem pi mu nu epsilon =
    let transport_cost = Tensor.sum' (Tensor.mul (C.cost mu nu) pi) |> Tensor.item in
    let regularization = epsilon *. (kl_divergence pi (Tensor.mm mu nu)) in
    transport_cost +. regularization

  let dual_problem u v mu nu epsilon =
    let obj = Tensor.sum' (Tensor.mul u mu) +. Tensor.sum' (Tensor.mul v nu) in
    let reg = epsilon *. Tensor.sum' (Tensor.exp ((Tensor.add u v -. C.cost mu nu) /. epsilon)) in
    obj -. reg |> Tensor.item

  let semi_dual_problem v mu nu epsilon =
    let v_c_eps x =
      if Float.(epsilon = 0.) then
        Tensor.min' (Tensor.sub (C.cost x nu) (Tensor.expand_as v x)) |> fst
      else
        let exp_term = Tensor.exp ((v -. C.cost x nu) /. epsilon) in
        -1. *. epsilon *. Tensor.log (Tensor.sum' exp_term)
    in
    let obj = Tensor.sum' (Tensor.mul (v_c_eps mu) mu) +. Tensor.sum' (Tensor.mul v nu) in
    obj -. epsilon |> Tensor.item

  let adaptive_sag_discrete_ot c mu nu epsilon max_iter =
    let n = Tensor.shape c |> List.hd_exn in
    let v = Tensor.zeros [n] in
    let d = Tensor.zeros [n] in
    let g = Tensor.zeros [n; n] in
    let step_size = ref (1. /. epsilon) in
    
    for k = 1 to max_iter do
      let i = Random.int n in
      let g_i = Tensor.sub nu (Tensor.softmax (Tensor.div (Tensor.sub v (Tensor.get c i)) epsilon)) in
      d <- Tensor.sub d (Tensor.get g i);
      Tensor.set g i g_i;
      d <- Tensor.add d g_i;
      
      let grad_norm = Tensor.norm d in
      step_size := Float.min (!step_size) (1. /. (grad_norm |> Tensor.item));
      
      v <- Tensor.add v (Tensor.mul_scalar d !step_size)
    done;
    v

  let minibatch_sgd_semi_discrete_ot mu nu c epsilon max_iter learning_rate batch_size =
    let n = Tensor.shape nu |> List.hd_exn in
    let v = Tensor.zeros [n] in
    let v_avg = Tensor.zeros [n] in

    for k = 1 to max_iter do
      let x_batch = mu batch_size in
      let grad_batch = Tensor.mean (Tensor.map (fun x_k ->
        Tensor.sub nu (Tensor.softmax (Tensor.div (Tensor.sub v (c x_k)) epsilon))
      ) x_batch) in
      let step_size = learning_rate /. Float.sqrt (Float.of_int k) in
      v <- Tensor.add v (Tensor.mul_scalar grad_batch step_size);
      v_avg <- Tensor.add (Tensor.mul_scalar v_avg ((Float.of_int (k - 1)) /. Float.of_int k))
                          (Tensor.mul_scalar v (1. /. Float.of_int k));
    done;
    v_avg
end

module Squared_euclidean_cost = struct
  let cost x y =
    let diff = Tensor.sub x y in
    Tensor.sum' (Tensor.mul diff diff) |> Tensor.reshape ~shape:[-1; 1]
end

module OT_squared_euclidean = OT(Squared_euclidean_cost)