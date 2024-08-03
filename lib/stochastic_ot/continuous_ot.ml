open Base
open Torch

type kernel = Tensor.t -> Tensor.t -> Tensor.t

exception Invalid_input of string

let gaussian_kernel sigma x x' =
  if Float.(sigma <= 0.) then raise (Invalid_input "sigma must be positive");
  let diff = Tensor.sub x x' in
  Tensor.exp (Tensor.mul_scalar (Tensor.norm diff) (-1. /. (2. *. sigma *. sigma)))

let kernel_sgd_continuous_ot mu nu c epsilon max_iter learning_rate kernel_x kernel_y =
  if Float.(epsilon <= 0.) then raise (Invalid_input "epsilon must be positive");
  if Int.(max_iter <= 0) then raise (Invalid_input "max_iter must be positive");
  if Float.(learning_rate <= 0.) then raise (Invalid_input "learning_rate must be positive");

  let alpha = ref [] in
  let x_samples = ref [] in
  let y_samples = ref [] in

  let sample_from_mu () = mu () in
  let sample_from_nu () = nu () in

  for k = 1 to max_iter do
    let x_k = sample_from_mu () in
    let y_k = sample_from_nu () in
    
    let u_k_1 = List.fold2_exn !alpha !x_samples ~init:(Tensor.zeros [1]) ~f:(fun acc a x -> 
      Tensor.add acc (Tensor.mul_scalar (kernel_x x x_k) a)) in
    
    let v_k_1 = List.fold2_exn !alpha !y_samples ~init:(Tensor.zeros [1]) ~f:(fun acc a y -> 
      Tensor.add acc (Tensor.mul_scalar (kernel_y y y_k) a)) in

    let new_alpha = learning_rate /. Float.sqrt (Float.of_int k) *. 
      (1. -. Float.exp ((Tensor.item u_k_1 +. Tensor.item v_k_1 -. c x_k y_k) /. epsilon)) in

    alpha := new_alpha :: !alpha;
    x_samples := x_k :: !x_samples;
    y_samples := y_k :: !y_samples;
  done;

  (List.rev !alpha, List.rev !x_samples, List.rev !y_samples)

let adaptive_kernel_sgd_continuous_ot mu nu c epsilon max_iter initial_lr kernel_x kernel_y =
  if Float.(epsilon <= 0.) then raise (Invalid_input "epsilon must be positive");
  if Int.(max_iter <= 0) then raise (Invalid_input "max_iter must be positive");
  if Float.(initial_lr <= 0.) then raise (Invalid_input "initial_lr must be positive");

  let alpha = ref [] in
  let x_samples = ref [] in
  let y_samples = ref [] in
  let learning_rate = ref initial_lr in

  let sample_from_mu () = mu () in
  let sample_from_nu () = nu () in

  for k = 1 to max_iter do
    let x_k = sample_from_mu () in
    let y_k = sample_from_nu () in
    
    let u_k_1 = List.fold2_exn !alpha !x_samples ~init:(Tensor.zeros [1]) ~f:(fun acc a x -> 
      Tensor.add acc (Tensor.mul_scalar (kernel_x x x_k) a)) in
    
    let v_k_1 = List.fold2_exn !alpha !y_samples ~init:(Tensor.zeros [1]) ~f:(fun acc a y -> 
      Tensor.add acc (Tensor.mul_scalar (kernel_y y y_k) a)) in

    let grad = 1. -. Float.exp ((Tensor.item u_k_1 +. Tensor.item v_k_1 -. c x_k y_k) /. epsilon) in
    
    learning_rate := !learning_rate /. Float.sqrt (Float.of_int k);
    
    let new_alpha = !learning_rate *. grad in

    alpha := new_alpha :: !alpha;
    x_samples := x_k :: !x_samples;
    y_samples := y_k :: !y_samples;
  done;

  (List.rev !alpha, List.rev !x_samples, List.rev !y_samples)