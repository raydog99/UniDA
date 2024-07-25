open Torch
open Subspace_robust_wasserstein
open Entropy_ot

let frank_wolfe_step (v_pi : Tensor.t) (k : int) : Tensor.t =
  let eigenvalues, eigenvectors = Tensor.symeig v_pi ~eigenvectors:true in
  let sorted_indices = Tensor.argsort eigenvalues ~descending:true in
  let top_k_indices = Tensor.narrow sorted_indices ~dim:0 ~start:0 ~length:k in
  let top_k_eigenvectors = Tensor.index_select eigenvectors ~dim:1 ~index:top_k_indices in
  Tensor.matmul top_k_eigenvectors (Tensor.transpose top_k_eigenvectors ~dim0:1 ~dim1:1)

let frank_wolfe_algorithm (x : Tensor.t) (y : Tensor.t) (k : int) (epsilon : float) (max_iter : int) (tol : float) : Tensor.t =
  let n, d = Tensor.shape2_exn x in
  let m, _ = Tensor.shape2_exn y in
  
  let cost_fn = SubspaceRobustWasserstein.cost_function in
  let pi = Entropy_ot.entropy_regularized_ot x y cost_fn epsilon max_iter in
  let v_pi = SubspaceRobustWasserstein.displacement_matrix x y pi in
  
  let omega = frank_wolfe_step v_pi k in
  
  let rec loop t omega =
    if t >= max_iter then omega
    else
      let pi = Entropy_ot.entropy_regularized_ot x y (fun x y -> cost_fn x y |> Tensor.matmul omega) epsilon max_iter in
      let v_pi = SubspaceRobustWasserstein.displacement_matrix x y pi in
      let omega_hat = frank_wolfe_step v_pi k in
      
      let tau = 2.0 /. (Float.of_int (t + 2)) in
      let new_omega = Tensor.((1.0 - tau) * omega + tau * omega_hat) in
      
      let duality_gap = Tensor.(sum (omega_hat * v_pi) - sum (omega * v_pi)) in
      if Tensor.float_value duality_gap < tol then new_omega
      else loop (t + 1) new_omega
  in
  
  loop 0 omega