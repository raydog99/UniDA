open Torch
open Types
open Lwt

let check_dataset_validity datasets =
  let s = Array.length datasets in
  if s = 0 then raise (HiwaError "Empty dataset");
  let d = Tensor.shape datasets.(0) |> snd in
  Array.iter (fun cluster ->
    let _, d' = Tensor.shape cluster in
    if d' <> d then raise (HiwaError "Inconsistent cluster dimensions")
  ) datasets

let parallel_map ~num_workers f arr =
  let chunks = Array.to_list arr |> List.map (fun x -> Lwt.return x) in
  Lwt_main.run (Lwt_list.map_p ~max_concurrency:num_workers f chunks)

let frobenius_norm t =
  Tensor.square t |> Tensor.sum |> Tensor.sqrt |> Tensor.item

let log level msg =
  Logs.msg level (fun m -> m "%s" msg)