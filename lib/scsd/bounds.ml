open Torch

let approximation_error_bound epsilon d l diameter =
  let open Float in
  2.0 *. epsilon *. float d *. log (exp 2.0 *. l *. diameter *. sqrt (float d) /. epsilon)

let sample_complexity_bound epsilon d l diameter n delta =
  let open Float in
  let sigma = 2.0 *. l *. diameter +. norm infinity in
  let b = 1.0 +. exp (2.0 *. sigma /. epsilon) in
  let k = exp (sigma /. epsilon) in
  
  6.0 *. b *. k *. sqrt (2.0 *. log (1.0 /. delta) /. float n) *.
  (max 1.0 (1.0 /. epsilon ** (float d /. 2.0)))

let estimate_lipschitz_constant samples =
  let n = Tensor.shape samples |> List.hd in
  let pairwise_distances = Tensor.(
    let x = unsqueeze samples ~dim:1 in
    let y = unsqueeze samples ~dim:0 in
    pow_scalar (sub x y) 2.0 |> sum ~dim:[2] |> sqrt
  ) in
  let max_distance = Tensor.max pairwise_distances |> Tensor.to_float0_exn in
  max_distance /. (2.0 *. sqrt (float n))

let estimate_diameter samples =
  let pairwise_distances = Tensor.(
    let x = unsqueeze samples ~dim:1 in
    let y = unsqueeze samples ~dim:0 in
    pow_scalar (sub x y) 2.0 |> sum ~dim:[2] |> sqrt
  ) in
  Tensor.max pairwise_distances |> Tensor.to_float0_exn