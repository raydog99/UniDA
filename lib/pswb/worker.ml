open Torch

module Make (C : Pswb.Config) = struct
  let compute_gradient x v =
    let open Tensor in
    let distances = sub (expand_as ~implicit:false x v) v in
    let i_w = argmin ~dim:(-1) ~keepdim:false distances in
    let grad = zeros_like v in
    scatter_add_ grad 0 i_w (full ~size:[C.num_points] 1.);
    grad

  let update v s x =
    let open Tensor in
    let grad = compute_gradient x v in
    let v' = add v (mul_scalar grad C.step_size) in
    let i_m = argmin s in
    let s' = add s (mul_scalar grad (C.step_size /. float C.num_workers)) in
    v', s', i_m, grad

  let worker_loop v s sample_fn =
    let rec loop iter v s =
      if iter >= C.max_iterations then Lwt.return ()
      else
        let x = sample_fn () in
        let v', s', i_m, grad = update v s x in
        Lwt_io.printf "Worker iteration %d\n" iter >>= fun () ->
        loop (iter + 1) v' s'
    in
    loop 0 v s
end