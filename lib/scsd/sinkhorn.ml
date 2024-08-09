open Torch

type measure = {
  samples: Tensor.t;
  weights: Tensor.t;
}

let create_measure samples weights =
  { samples; weights }

let uniform_measure samples =
  let n = Tensor.shape samples |> List.hd in
  let weights = Tensor.ones [n] ~device:(Tensor.device samples) in
  create_measure samples weights

let quadratic_cost x y =
  Tensor.(pow_scalar (sub x y) 2.0 |> sum ~dim:[1] |> unsqueeze ~dim:1)

let l1_cost x y =
  Tensor.(abs (sub x y) |> sum ~dim:[1] |> unsqueeze ~dim:1)

let sinkhorn alpha beta cost epsilon max_iter =
  let n = Tensor.shape alpha.samples |> List.hd in
  let m = Tensor.shape beta.samples |> List.hd in
  let K = Tensor.(exp (neg (div_scalar cost epsilon))) in
  let u = Tensor.ones [n; 1] ~device:(Tensor.device alpha.samples) in
  let v = Tensor.ones [m; 1] ~device:(Tensor.device beta.samples) in
  
  let rec iterate i u v =
    if i >= max_iter then (u, v)
    else
      let u' = Tensor.(div alpha.weights (matmul K v)) in
      let v' = Tensor.(div beta.weights (matmul (transpose K ~dim0:0 ~dim1:1) u')) in
      iterate (i + 1) u' v'
  in
  
  iterate 0 u v

let sinkhorn_divergence alpha beta cost epsilon max_iter =
  let (u_ab, v_ab) = sinkhorn alpha beta cost epsilon max_iter in
  let (u_aa, v_aa) = sinkhorn alpha alpha cost epsilon max_iter in
  let (u_bb, v_bb) = sinkhorn beta beta cost epsilon max_iter in
  
  let w_ab = Tensor.(sum (mul (mul u_ab (matmul cost v_ab)) alpha.weights)) in
  let w_aa = Tensor.(sum (mul (mul u_aa (matmul cost v_aa)) alpha.weights)) in
  let w_bb = Tensor.(sum (mul (mul u_bb (matmul cost v_bb)) beta.weights)) in
  
  Tensor.(sub w_ab (div_scalar (add w_aa w_bb) 2.0))

let sample_complexity_experiment n_samples d epsilon n_runs cost_fn =
  let device = Device.cuda_if_available () in
  let alpha = Tensor.randn [n_samples; d] ~device in
  let beta = Tensor.randn [n_samples; d] ~device in
  
  let alpha_measure = uniform_measure alpha in
  let beta_measure = uniform_measure beta in
  
  let cost = cost_fn alpha beta in
  
  let divergences = List.init n_runs (fun _ ->
    sinkhorn_divergence alpha_measure beta_measure cost epsilon 100
  ) in
  
  let mean_divergence = Tensor.(mean (stack divergences ~dim:0)) in
  let std_divergence = Tensor.(std (stack divergences ~dim:0) ~unbiased:true) in
  
  (Tensor.to_float0_exn mean_divergence, Tensor.to_float0_exn std_divergence)