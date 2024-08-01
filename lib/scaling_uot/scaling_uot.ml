open Torch
open Divergences
open Wasserstein_fisher_rao
open Uot_utils
open Numerical_utils

type cost = Tensor.t
type kernel = Tensor.t
type scaling = Tensor.t
type marginal = Tensor.t

type error =
  | InvalidDimension
  | NegativeValues
  | DivergenceError of string
  | MaxIterationsReached
  | NumericalInstability

let compute_kernel (c : cost) (epsilon : float) : kernel =
  Tensor.(exp (neg c / f epsilon))

let scaling_iteration 
    (k : kernel) 
    (a : scaling) 
    (b : scaling) 
    (mu : marginal) 
    (nu : marginal) 
    (epsilon : float)
    (div1 : divergence)
    (div2 : divergence) : (scaling * scaling, error) result =
  try
    let open Tensor in
    let kb = matmul k b in
    let a_new = proximal_kl (fun x -> div1 x mu) kb epsilon in
    let ka_new = matmul (transpose k) a_new in
    let b_new = proximal_kl (fun x -> div2 x nu) ka_new epsilon in
    let a_new_stable = stabilize_tensor a_new epsilon in
    let b_new_stable = stabilize_tensor b_new epsilon in
    Ok (a_new_stable, b_new_stable)
  with
  | _ -> Error NumericalInstability

let scaling_algorithm 
    (c : cost) 
    (mu : marginal) 
    (nu : marginal) 
    (epsilon : float) 
    (max_iter : int) 
    (tol : float)
    (div1 : divergence)
    (div2 : divergence) : (scaling * scaling, error) result =
  if Tensor.shape c <> [Tensor.shape mu |> List.hd; Tensor.shape nu |> List.hd] then
    Error InvalidDimension
  else if Tensor.(min c |> to_float0_exn) < 0.0 then
    Error NegativeValues
  else if not (is_probability_vector mu && is_probability_vector nu) then
    Error (DivergenceError "Marginals must be probability vectors")
  else
    let k = compute_kernel c epsilon in
    let a = Tensor.ones [Tensor.shape mu |> List.hd] in
    let b = Tensor.ones [Tensor.shape nu |> List.hd] in
    
    let rec iterate a b iter =
      if iter >= max_iter then Error MaxIterationsReached
      else
        match scaling_iteration k a b mu nu epsilon div1 div2 with
        | Error e -> Error e
        | Ok (a_new, b_new) ->
            let err = Tensor.((a_new - a) / a |> abs |> max) |> Tensor.to_float0_exn in
            if err < tol then Ok (a_new, b_new)
            else iterate a_new b_new (iter + 1)
    in
    
    iterate a b 0

let compute_transport_plan 
    (a : scaling) 
    (k : kernel) 
    (b : scaling) : Tensor.t =
  Tensor.(a * (matmul k (b * b)) * a)

let wasserstein_distance 
    (c : cost) 
    (pi : Tensor.t) : float =
  Tensor.(sum (c * pi)) |> Tensor.to_float0_exn

let wasserstein_fisher_rao_distance 
    (mu : marginal) 
    (nu : marginal) 
    (c : cost) 
    (epsilon : float) : (float, error) result =
  let lambda = epsilon /. 2. in
  let wfr_c = wfr_cost c lambda in
  match scaling_algorithm wfr_c mu nu epsilon 1000 1e-6 kl_divergence kl_divergence with
  | Error e -> Error e
  | Ok (a, b) ->
      let pi = compute_transport_plan a (compute_kernel wfr_c epsilon) b in
      Ok (wfr_distance mu nu pi c epsilon)

let gaussian_hellinger_kantorovich_distance 
    (mu : marginal) 
    (nu : marginal) 
    (c : cost) 
    (epsilon : float)
    (lambda : float) : (float, error) result =
  let ghk_div p q = ghk_divergence lambda p q in
  match scaling_algorithm c mu nu epsilon 1000 1e-6 ghk_div ghk_div with
  | Error e -> Error e
  | Ok (a, b) ->
      let pi = compute_transport_plan a (compute_kernel c epsilon) b in
      Ok (wasserstein_distance c pi)

let entropy_regularized_ot 
    (c : cost) 
    (mu : marginal) 
    (nu : marginal) 
    (epsilon : float) : (Tensor.t, error) result =
  match scaling_algorithm c mu nu epsilon 1000 1e-6 kl_divergence kl_divergence with
  | Error e -> Error e
  | Ok (a, b) ->
      let k = compute_kernel c epsilon in
      let pi = compute_transport_plan a k b in
      Ok pi

let print_transport_plan (pi : Tensor.t) : unit =
  print_endline "Optimal Transport Plan:";
  print_tensor pi

let print_distance (distance : float) : unit =
  Printf.printf "Distance: %f\n" distance

let example_usage () =
  let mu = generate_random_marginal 5 in
  let nu = generate_random_marginal 5 in
  let c = generate_random_cost_matrix 5 5 in
  let epsilon = 0.1 in
  let max_iter = 1000 in
  let tol = 1e-6 in

  match scaling_algorithm c mu nu epsilon max_iter tol kl_divergence kl_divergence with
  | Error e -> 
      (match e with
      | InvalidDimension -> print_endline "Error: Invalid dimensions"
      | NegativeValues -> print_endline "Error: Negative values in cost matrix"
      | DivergenceError msg -> Printf.printf "Error in divergence: %s\n" msg
      | MaxIterationsReached -> print_endline "Error: Maximum iterations reached"
      | NumericalInstability -> print_endline "Error: Numerical instability occurred")
  | Ok (a, b) ->
      let k = compute_kernel c epsilon in
      let pi = compute_transport_plan a k b in
      let distance = wasserstein_distance c pi in
      print_endline "Optimal Transport Plan:";
      print_tensor pi;
      Printf.printf "Wasserstein distance: %f\n" distance

let wfr_example () =
  let mu = generate_random_marginal 5 in
  let nu = generate_random_marginal 5 in
  let c = generate_random_cost_matrix 5 5 in
  let epsilon = 0.1 in

  match wasserstein_fisher_rao_distance mu nu c epsilon with
  | Error e -> 
      (match e with
      | InvalidDimension -> print_endline "Error: Invalid dimensions"
      | NegativeValues -> print_endline "Error: Negative values in cost matrix"
      | DivergenceError msg -> Printf.printf "Error in divergence: %s\n" msg
      | MaxIterationsReached -> print_endline "Error: Maximum iterations reached"
      | NumericalInstability -> print_endline "Error: Numerical instability occurred")
  | Ok distance ->
      Printf.printf "Wasserstein-Fisher-Rao distance: %f\n" distance

let ghk_example () =
  let mu = generate_random_marginal 5 in
  let nu = generate_random_marginal 5 in
  let c = generate_random_cost_matrix 5 5 in
  let epsilon = 0.1 in
  let lambda = 1.0 in

  match gaussian_hellinger_kantorovich_distance mu nu c epsilon lambda with
  | Error e -> 
      (match e with
      | InvalidDimension -> print_endline "Error: Invalid dimensions"
      | NegativeValues -> print_endline "Error: Negative values in cost matrix"
      | DivergenceError msg -> Printf.printf "Error in divergence: %s\n" msg
      | MaxIterationsReached -> print_endline "Error: Maximum iterations reached"
      | NumericalInstability -> print_endline "Error: Numerical instability occurred")
  | Ok distance ->
      Printf.printf "Gaussian-Hellinger-Kantorovich distance: %f\n" distance