open Torch
open Types
open Utils
open Lwt
open Config

let compute_cost r x y =
  let rx = Tensor.mm r x in
  Tensor.mean (Tensor.square (Tensor.sub rx y))

let update_r_ij p_ij y_j q_ij x_i r lambda_ij mu =
  let term1 = Tensor.mul_scalar (Tensor.mm y_j (Tensor.transpose q_ij 0 1)) (2. *. p_ij) in
  let term2 = Tensor.mm (Tensor.transpose x_i 0 1) term1 in
  let term3 = Tensor.mul_scalar (Tensor.sub r lambda_ij) mu in
  Stiefel.stiefel_alignment (Tensor.add term2 term3)

let update_q_ij epsilon2 p_ij r_ij x_i y_j max_iter tol =
  let cost_matrix = Tensor.square (Tensor.sub (Tensor.mm r_ij x_i) y_j) in
  Sinkhorn.sinkhorn (epsilon2 /. p_ij) cost_matrix max_iter tol

let update_p epsilon1 costs max_iter tol =
  Sinkhorn.sinkhorn epsilon1 costs max_iter tol

let update_cluster_pair config i j r p r_ij q_ij lambda_ij x y =
  let p_ij = Tensor.get p [i; j] in
  let r_ij' = update_r_ij p_ij y.(j) q_ij.(i).(j) x.(i) r lambda_ij.(i).(j) (get_mu config) in
  let q_ij' = update_q_ij (get_epsilon2 config) p_ij r_ij' x.(i) y.(j) (get_max_iter config) (get_tolerance config) in
  (r_ij', q_ij')

let hiwa config x y =
  check_dataset_validity x;
  check_dataset_validity y;
  let s = Array.length x in
  let d = Tensor.shape x.(0) |> snd in
  
  let r = Tensor.randn [d; d] |> Stiefel.project_stiefel in
  let p = Tensor.full [s; s] (1. /. float_of_int (s * s)) in
  let r_ij = Array.make_matrix s s r in
  let q_ij = Array.make_matrix s s (Tensor.empty [0]) in
  let lambda_ij = Array.make_matrix s s (Tensor.zeros [d; d]) in
  
  let rec iterate r p r_ij q_ij lambda_ij iter =
    if iter >= get_max_iter config then r
    else
      (* Update R_ij and Q_ij in parallel *)
      let updates = parallel_map ~num_workers:(get_num_workers config)
        (fun (i, j) -> update_cluster_pair config i j r p r_ij q_ij lambda_ij x y)
        (Array.init (s * s) (fun k -> (k / s, k mod s))) in
      Array.iteri (fun k (r_ij', q_ij') ->
        let i, j = k / s, k mod s in
        r_ij.(i).(j) <- r_ij';
        q_ij.(i).(j) <- q_ij'
      ) updates;
      
      (* Update P *)
      let costs = Tensor.of_float2 (Array.make_matrix s s 0.) in
      for i = 0 to s - 1 do
        for j = 0 to s - 1 do
          let cost = compute_cost r_ij.(i).(j) x.(i) y.(j) in
          Tensor.set costs [i; j] cost
        done
      done;
      let p' = update_p (get_epsilon1 config) costs (get_max_iter config) (get_tolerance config) in
      
      (* Update R *)
      let r' = Tensor.zeros [d; d] in
      for i = 0 to s - 1 do
        for j = 0 to s - 1 do
          r' <- Tensor.add r' (Tensor.add r_ij.(i).(j) lambda_ij.(i).(j))
        done
      done;
      let r' = Stiefel.stiefel_alignment r' in
      
      (* Update Lambda_ij *)
      for i = 0 to s - 1 do
        for j = 0 to s - 1 do
          lambda_ij.(i).(j) <- Tensor.add lambda_ij.(i).(j) (Tensor.sub r_ij.(i).(j) r')
        done
      done;
      
      let diff = Tensor.sub r' r |> frobenius_norm in
      log (get_log_level config) (Printf.sprintf "Iteration %d: diff = %f" iter diff);
      if diff < get_tolerance config then r'
      else iterate r' p' r_ij q_ij lambda_ij (iter + 1)
  in
  
  iterate r p r_ij q_ij lambda_ij 0