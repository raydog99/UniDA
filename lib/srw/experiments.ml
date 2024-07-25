open Torch
open Subspace_robust_wasserstein
open Utils

(** Run the fragmented hypercube experiment *)
let fragmented_hypercube_experiment (d : int) (k_star : int) (n : int) (epsilon : float) (max_iter : int) (tol : float) : unit =
  let mu, nu = Utils.generate_fragmented_hypercube d k_star n in
  let distances = SubspaceRobustWasserstein.srw_for_all_k mu nu epsilon max_iter tol in
  Printf.printf "Fragmented Hypercube (d=%d, k*=%d):\n" d k_star;
  List.iter (fun (k, dist) -> Printf.printf "k=%d: %f\n" k dist) distances

(** Run the noise robustness experiment *)
let noise_robustness_experiment (d : int) (k_star : int) (n : int) (epsilon : float) (max_iter : int) (tol : float) (noise_levels : float list) : unit =
  let mu, nu = Utils.generate_fragmented_hypercube d k_star n in
  List.iter (fun noise_level ->
    let noisy_nu = Tensor.(nu + (randn (shape nu) * noise_level)) in
    let distances = SubspaceRobustWasserstein.srw_for_all_k mu noisy_nu epsilon max_iter tol in
    Printf.printf "Noise Robustness (d=%d, k*=%d, noise=%.2f):\n" d k_star noise_level;
    List.iter (fun (k, dist) -> Printf.printf "k=%d: %f\n" k dist) distances
  ) noise_levels

(** Run the elbow method experiment *)
let elbow_method_experiment (d : int) (k_star : int) (n : int) (epsilon : float) (max_iter : int) (tol : float) (threshold : float) : unit =
  let mu, nu = Utils.generate_fragmented_hypercube d k_star n in
  let chosen_k = SubspaceRobustWasserstein.choose_k_elbow mu nu epsilon max_iter tol threshold in
  Printf.printf "Elbow Method (d=%d, k*=%d): Chosen k=%d\n" d k_star chosen_k