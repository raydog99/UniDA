open Torch
open Wasserstein_distance
open Empirical_measure
open Linear_program_solver

let approximate_wasserstein_distance r s sample_size num_repetitions p =
let results = Array.init num_repetitions (fun _ ->
  let r_empirical = Empirical_measure.sample_empirical_measure r sample_size in
  let s_empirical = Empirical_measure.sample_empirical_measure s sample_size in
  Wasserstein_distance.compute_wasserstein_distance r_empirical s_empirical p
) in

let sum = Array.fold_left (+.) 0. results in
sum /. float_of_int num_repetitions

let compute_error_bound n d p s =
let open Float in
let e_q = 2. ** (1. -. 1. /. (2. *. p)) *. (float_of_int d) ** (p /. 2.) *. 
          (float_of_int n) ** (1. /. (2. *. p)) in
2. *. e_q *. (1. /. sqrt (float_of_int s)) ** (1. /. p)

let compute_covering_number x epsilon =
let diameter = Wasserstein_distance.compute_diameter x in
let d = Tensor.shape x |> List.tl |> List.hd in
let volume_ratio = (diameter /. epsilon) ** float_of_int d in
ceil (volume_ratio *. log (float_of_int (Tensor.shape x |> List.hd)))

let compute_e_q x p =
let n = Tensor.shape x |> List.hd in
let d = Tensor.shape x |> List.tl |> List.hd in
let diameter = Wasserstein_distance.compute_diameter x in
let q = 2 in
let l_max = int_of_float (log (float_of_int n) /. log (float_of_int q)) in

let sum_term = 
  List.init (l_max + 1) (fun l ->
    let epsilon = diameter *. (float_of_int q ** float_of_int (-l)) in
    let covering_number = compute_covering_number x epsilon in
    (float_of_int q ** (float_of_int l *. p)) *. sqrt covering_number
  ) |> List.fold_left (+.) 0.
in

2. ** (1. -. 1. /. (2. *. p)) *. diameter ** p *. 
(float_of_int q ** (float_of_int (l_max + 1) *. p) *. sqrt (float_of_int n) +. sum_term)