open Torch

module Make (C : Pswb.Config) = struct
  let init_tensors () =
    let open Tensor in
    let s = zeros [C.num_points] in
    let v = List.init C.num_workers (fun _ -> zeros [C.num_points]) in
    s, v

  let update s i_m i_w =
    let open Tensor in
    let s' = add s (float_vec [|C.step_size /. (2. *. float C.num_workers)|]) in
    let s'' = sub s' (float_vec [|C.step_size /. 2.|]) in
    s''

  let master_loop s v sample_fns =
    let rec loop iter s =
      if iter >= C.max_iterations then Lwt.return ()
      else
        let workers = List.mapi (fun i v_i ->
          Worker.Make(C).worker_loop v_i s (List.nth sample_fns i)
        ) v in
        Lwt.join workers >>= fun () ->
        Lwt_io.printf "Master iteration %d\n" iter >>= fun () ->
        loop (iter + 1) s
    in
    loop 0 s

  let run sample_fns =
    let s, v = init_tensors () in
    master_loop s v sample_fns
end