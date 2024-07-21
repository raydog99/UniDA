open Torch

type cost_matrix = Tensor.t

let extend_cost_matrix c xi a =
  let n, m = Tensor.shape2_exn c in
  let extended = Tensor.zeros [n+1; m+1] in
  let _ = Tensor.copy_ ~src:c ~dst:(Tensor.narrow extended ~dim:0 ~start:0 ~length:n |> Tensor.narrow ~dim:1 ~start:0 ~length:m) in
  let _ = Tensor.fill_ (Tensor.narrow extended ~dim:0 ~start:n ~length:1) xi in
  let _ = Tensor.fill_ (Tensor.narrow extended ~dim:1 ~start:m ~length:1) xi in
  Tensor.set_ extended [|n; m|] (Scalar.float (2. *. xi +. a));
  extended

let partial_wasserstein c p q s =
  let extended_c = extend_cost_matrix c 0. (Tensor.max c |> Tensor.item)  in
  let extended_p = Tensor.cat [p; Tensor.full [1] (Tensor.sum q |> Tensor.item)] ~dim:0 in
  let extended_q = Tensor.cat [q; Tensor.full [1] (Tensor.sum p |> Tensor.item)] ~dim:0 in
  Utils.solve_linear_program extended_c extended_p extended_q

let compute_partial_gw_loss cs ct t =
  let n, m = Tensor.shape2_exn t in
  let loss = Tensor.zeros [] in
  for i = 0 to n - 1 do
    for j = 0 to m - 1 do
      for k = 0 to n - 1 do
        for l = 0 to m - 1 do
          let diff = Tensor.get cs [|i; k|] -. Tensor.get ct [|j; l|] in
          let contrib = diff *. diff *. Tensor.get t [|i; j|] *. Tensor.get t [|k; l|] in
          Tensor.add_ loss contrib
        done
      done
    done
  done;
  Tensor.item loss /. 2.

let gradient_partial_gw cs ct t =
  let n, m = Tensor.shape2_exn t in
  let grad = Tensor.zeros [n; m] in
  for i = 0 to n - 1 do
    for j = 0 to m - 1 do
      let sum = ref 0. in
      for k = 0 to n - 1 do
        for l = 0 to m - 1 do
          let diff = Tensor.get cs [|i; k|] -. Tensor.get ct [|j; l|] in
          sum := !sum +. (diff *. diff *. Tensor.get t [|k; l|])
        done
      done;
      Tensor.set_ grad [|i; j|] (Scalar.float !sum)
    done
  done;
  grad

let partial_gromov_wasserstein cs ct p q s =
  Frank_Wolfe.optimize (Frank_Wolfe.partial_gw_step cs ct p q s) (Tensor.zeros [Tensor.shape p; Tensor.shape q]) 100 1e-6 None