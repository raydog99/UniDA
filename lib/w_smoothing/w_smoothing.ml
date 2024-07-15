open Torch

let laplace_noise shape lambda =
  let u1 = Tensor.rand shape ~kind:Float in
  let u2 = Tensor.rand shape ~kind:Float in
  Tensor.(log (u1 / (Scalar.float 1. - u1)) * (Scalar.float lambda / (Scalar.float 2.)))

let apply_flow x flow =
  let n, m = Tensor.shape2_exn x in
  let flow_vert, flow_horiz = Tensor.split ~dim:0 flow [n-1; n] in
  let flow_vert = Tensor.pad flow_vert ~pad:[0;0;1;1;0;0] ~mode:"constant" ~value:0. in
  let flow_horiz = Tensor.pad flow_horiz ~pad:[0;0;0;0;1;1] ~mode:"constant" ~value:0. in
  Tensor.(x + (flow_vert.(.., 0) - flow_vert.(.., 1)) + (flow_horiz.(0, ..) - flow_horiz.(1, ..)))

let smooth classifier lambda x =
  let n, m = Tensor.shape2_exn x in
  let flow_shape = [2 * n - 1; m] in
  let noise = laplace_noise flow_shape lambda in
  classifier (apply_flow x noise)

let certify classifier lambda x radius =
  let n_samples = 10000 in
  let samples = List.init n_samples (fun _ -> smooth classifier lambda x) in
  let counts = Hashtbl.create 10 in
  List.iter (fun pred ->
    let class_id = Tensor.argmax pred ~dim:0 ~keepdim:false in
    let count = Hashtbl.find_opt counts class_id |> Option.value ~default:0 in
    Hashtbl.replace counts class_id (count + 1)
  ) samples;
  let top_class, top_count = Hashtbl.fold (fun k v (mk, mv) ->
    if v > mv then (k, v) else (mk, mv)
  ) counts (-1, 0) in
  let p_a = float_of_int top_count /. float_of_int n_samples in
  let certified_radius = lambda *. log (p_a /. (1. -. p_a)) /. (4. *. radius) in
  (certified_radius, if certified_radius >= radius then Some top_class else None)