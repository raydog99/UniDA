open Torch

let create () =
  let model = 
    Nn.sequential
      [ Nn.linear ~in_features:784 ~out_features:128 ()
      ; Nn.relu ()
      ; Nn.linear ~in_features:128 ~out_features:10 ()
      ]
  in
  fun x -> Nn.Module.forward model x

let train classifier x_train y_train =
  let optimizer = Optimizer.adam (Nn.Module.parameters classifier) ~lr:0.01 in
  for epoch = 1 to 10 do
    let loss = 
      Tensor.(mean (nll_loss (classifier x_train) y_train))
    in
    Optimizer.backward_step optimizer ~loss;
    Stdio.printf "Epoch %d, Loss: %f\n" epoch (Tensor.float_value loss)
  done

let evaluate classifier x_test y_test =
  let pred = classifier x_test in
  let accuracy = 
    Tensor.(sum (argmax pred ~dim:1 ~keepdim:false == y_test))
    |> Tensor.float_value
  in
  accuracy /. (float_of_int (Tensor.shape y_test).(0))