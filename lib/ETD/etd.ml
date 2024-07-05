open Torch

module ETD = struct
  type t = {
    w: Var_store.t;
    w_g: Var_store.t;
    w_epsilon: Var_store.t;
    c: Tensor.t;
    epsilon: float;
    lambda: float;
  }

  let create ~b ~epsilon ~lambda =
    let w = Var_store.create ~name:"w" () in
    let w_g = Var_store.create ~name:"w_g" () in
    let w_epsilon = Var_store.create ~name:"w_epsilon" () in
    let c = Tensor.randn [b; b] in
    { w; w_g; w_epsilon; c; epsilon; lambda }

  let pre_train t ~x_s ~y_s =
    let model = Layer.sequential [
      Layer.linear t.w ~input_dim:(Tensor.shape x_s).(1) ~output_dim:128;
      Layer.relu;
      Layer.linear t.w ~input_dim:128 ~output_dim:(Tensor.shape y_s).(1);
    ] in
    let optimizer = Optimizer.adam (Var_store.all_vars t.w) ~learning_rate:1e-3 in
    for _ = 1 to 1000 do
      let predicted = Layer.forward model x_s in
      let loss = Tensor.mse_loss predicted y_s in
      Optimizer.backward_step optimizer ~loss;
    done

  let predict_pseudo_labels t x_t =
    let model = Layer.sequential [
      Layer.linear t.w ~input_dim:(Tensor.shape x_t).(1) ~output_dim:128;
      Layer.relu;
      Layer.linear t.w ~input_dim:128 ~output_dim:(Tensor.shape x_t).(1);
    ] in
    Layer.forward model x_t

  let calculate_attention t s =
    let s_exp = Tensor.exp s in
    Tensor.div s_exp (Tensor.sum s_exp ~dim:[1] ~keepdim:true)

  let re_weigh_c t s =
    Tensor.mul s t.c

  let update_w_g t =
    let optimizer = Optimizer.adam (Var_store.all_vars t.w_g) ~learning_rate:1e-3 in
    for _ = 1 to 100 do
      let loss = Tensor.mse_loss (Layer.forward (Layer.linear t.w_g) t.w_epsilon) Tensor.zeros_like in
      Optimizer.backward_step optimizer ~loss;
    done

  let update_w t =
    let optimizer = Optimizer.adam (Var_store.all_vars t.w) ~learning_rate:1e-3 in
    for _ = 1 to 100 do
      let loss = Tensor.mse_loss (Layer.forward (Layer.linear t.w) t.w_epsilon) Tensor.zeros_like in
      Optimizer.backward_step optimizer ~loss;
    done

  let train t ~x_s ~y_s ~x_t =
    pre_train t ~x_s ~y_s;
    let y_t_hat = predict_pseudo_labels t x_t in
    let rec outer_loop () =
      let s = calculate_attention t (Tensor.matmul x_t (Tensor.transpose x_s ~dim0:0 ~dim1:1)) in
      let c_new = re_weigh_c t s in
      t.c <- c_new;
      let rec inner_loop () =
        update_w_g t;
        if Tensor.mean (Tensor.abs (Tensor.sub t.w_g t.w_epsilon)) > t.epsilon then
          inner_loop ()
      in
      inner_loop ();
      update_w t;
      if Tensor.mean (Tensor.abs (Tensor.sub t.w t.w_epsilon)) > t.lambda then
        outer_loop ()
    in
    outer_loop ();
    (t.w, t.w_g, t.w_epsilon, y_t_hat)
end