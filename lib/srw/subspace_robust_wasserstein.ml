open Torch
open Frank_wolfe
open Entropy_ot

let displacement_matrix (x : Tensor.t) (y : Tensor.t) (pi : Tensor.t) : Tensor.t =
  let diff = Tensor.(x - y) in
  Tensor.(matmul (transpose diff ~dim0:1 ~dim1:2) diff * pi.unsqueeze(-1))

let cost_function (x : Tensor.t) (y : Tensor.t) : Tensor.t =
  let x_exp = Tensor.unsqueeze x ~dim:1 in
  let y_exp = Tensor.unsqueeze y ~dim:0 in
  Tensor.(pow (x_exp - y_exp) (Scalar.f 2.0) |> sum ~dim:[-1])

let srw (mu : Tensor.t) (nu : Tensor.t) (k : int) (epsilon : float) (max_iter : int) (tol : float) : Tensor.t =
  let omega = Frank_wolfe.frank_wolfe_algorithm mu nu k epsilon max_iter tol in
  let pi = Entropy_ot.entropy_regularized_ot mu nu (fun x y -> cost_function x y |> Tensor.matmul omega) epsilon max_iter in
  let v_pi = displacement_matrix mu nu pi in
  Tensor.trace (Tensor.matmul omega v_pi)

let srw_for_all_k (mu : Tensor.t) (nu : Tensor.t) (epsilon : float) (max_iter : int) (tol : float) : (int * float) list =
  let _, d = Tensor.shape2_exn mu in
  List.init d (fun k -> 
    let k = k + 1 in
    let dist = srw mu nu k epsilon max_iter tol in
    (k, Tensor.float_value dist)
  )

let choose_k_elbow (mu : Tensor.t) (nu : Tensor.t) (epsilon : float) (max_iter : int) (tol : float) (threshold : float) : int =
  let distances = srw_for_all_k mu nu epsilon max_iter tol in
  let rec find_elbow prev_slope = function
    | [] | [_] -> List.length distances
    | (k1, d1) :: (k2, d2) :: rest ->
        let slope = (d2 -. d1) /. (float_of_int (k2 - k1)) in
        if abs_float (slope -. prev_slope) < threshold then k1
        else find_elbow slope ((k2, d2) :: rest)
  in
  match distances with
  | [] | [_] -> 1
  | (_, d1) :: (k2, d2) :: rest ->
      let initial_slope = (d2 -. d1) /. (float_of_int (k2 - 1)) in
      find_elbow initial_slope ((k2, d2) :: rest)