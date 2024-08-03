open Base
open Torch

exception Invalid_input of string

let sgd_semi_discrete_ot mu nu c epsilon max_iter learning_rate =
  let n = Tensor.shape nu |> List.hd_exn in
  if Int.(n = 0) then raise (Invalid_input "nu must not be empty");
  
  let v = Tensor.zeros [n] in
  let v_avg = Tensor.zeros [n] in

  let sample_from_mu () = mu () in

  for k = 1 to max_iter do
    let x_k = sample_from_mu () in
    let grad = Tensor.sub nu (Tensor.softmax (Tensor.div (Tensor.sub v (c x_k)) epsilon)) in
    let step_size = learning_rate /. Float.sqrt (Float.of_int k) in
    v <- Tensor.add v (Tensor.mul_scalar grad step_size);
    v_avg <- Tensor.add (Tensor.mul_scalar v_avg ((Float.of_int (k - 1)) /. Float.of_int k))
                        (Tensor.mul_scalar v (1. /. Float.of_int k));
  done;
  v_avg

let minibatch_sgd_semi_discrete_ot mu nu c epsilon max_iter learning_rate batch_size =
  let n = Tensor.shape nu |> List.hd_exn in
  if Int.(n = 0) then raise (Invalid_input "nu must not be empty");
  
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

let h_epsilon x v nu c epsilon =
  if Float.(epsilon < 0.) then raise (Invalid_input "epsilon must be non-negative");
  
  let v_c_eps =
    if Float.(epsilon = 0.) then
      Tensor.min' (Tensor.sub (c x) v) |> fst
    else
      let exp_term = Tensor.exp (Tensor.div (Tensor.sub v (c x)) epsilon) in
      Tensor.mul_scalar (Tensor.log (Tensor.sum' (Tensor.mul exp_term nu))) (-1. *. epsilon)
  in
  Tensor.add (Tensor.sum' (Tensor.mul v nu)) v_c_eps