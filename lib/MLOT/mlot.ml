open Torch

module MLOT = struct
  let create_linear dim =
    let w = Tensor.randn [dim; dim] ~requires_grad:true in
    let b = Tensor.zeros [dim] ~requires_grad:true in
    (w, b)

  let linear_forward w b x =
    Tensor.(matmul x (transpose w ~dim0:0 ~dim1:1) + b)

  let wasserstein_distance xs xt w b =
    let cs = linear_forward w b xs in
    let ct = linear_forward w b xt in
    let m = Tensor.shape cs |> List.hd in
    let n = Tensor.shape ct |> List.hd in
    let ot_matrix = Tensor.((ones [m; n]) / (Scalar.f (float_of_int (m * n)))) in
    let cost_matrix = Tensor.(
      (pow_tensor_scalar (cs.unsqueeze 1 - ct.unsqueeze 0) (Scalar.f 2.)).sum ~dim:[~-1]
    ) in
    Tensor.((ot_matrix * cost_matrix).sum())

  let entropy h p =
    let eps = 1e-8 in
    Tensor.(-(p * (log (p + Scalar.f eps))).sum())

  let loss xs xt ys yt w b lambda =
    let ws_dist = wasserstein_distance xs xt w b in
    let cs = linear_forward w b xs in
    let ct = linear_forward w b xt in
    let ps = Tensor.softmax cs ~dim:1 in
    let pt = Tensor.softmax ct ~dim:1 in
    let class_loss = Tensor.(
      (cross_entropy_loss ps ys ~reduction:Sum) +
      (cross_entropy_loss pt yt ~reduction:Sum)
    ) in
    let entropy_loss = Tensor.(
      (entropy (Scalar.f 1.) ps) +
      (entropy (Scalar.f 1.) pt)
    ) in
    Tensor.(ws_dist + (Scalar.f lambda * (class_loss - entropy_loss)))

  let train xs xt ys yt num_epochs learning_rate lambda =
    let dim = Tensor.shape xs |> List.hd in
    let w, b = create_linear dim in
    let optimizer = Optimizer.adam [w; b] ~lr:learning_rate in

    for _ = 1 to num_epochs do
      Optimizer.zero_grad optimizer;
      let l = loss xs xt ys yt w b lambda in
      backward l;
      Optimizer.step optimizer;
      Printf.printf "Loss: %f\n" (Tensor.to_float0_exn l)
    done;
    (w, b)

  let transform x w b =
    linear_forward w b x

  let predict x w b =
    let logits = transform x w b in
    Tensor.argmax logits ~dim:1 ~keepdim:false
end