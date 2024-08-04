open Torch
open Yojson.Safe
open Model
open Optimal_transport
open Config

type t = {
  mutable f: Tensor.t -> Tensor.t;
  mutable optimizer: Optimizer.t;
  config: Config.t;
  loss_fn: Loss.t;
}

let create model_type loss_fn config =
  let f, vs = Model.create model_type in
  let optimizer = Optimizer.adam (Var_store.vars vs) ~learning_rate:config.Config.learning_rate in
  { f; optimizer; config; loss_fn }

let compute_cost jdot xs ys xt ft =
  let d_xx = Tensor.(mean (sub xs xt |> pow_scalar 2.)) in
  let l_yy = Loss.compute jdot.loss_fn ys ft in
  Tensor.(add (mul_scalar d_xx jdot.config.Config.alpha) l_yy)

let train_step jdot xs ys xt =
  Optimizer.zero_grad jdot.optimizer;
  let ft = jdot.f xt in
  let cost_matrix = Optimal_transport.compute_cost_matrix xs xt ys ft jdot.config.Config.alpha in
  let ot_matrix = Optimal_transport.sinkhorn_knopp cost_matrix jdot.config.Config.epsilon jdot.config.Config.max_iter in
  let loss = compute_cost jdot xs ys xt ft in
  Tensor.backward loss;
  Optimizer.step jdot.optimizer;
  Tensor.to_float0_exn loss

let evaluate jdot data =
  let yt = Option.get data.Data.yt in
  let pred = jdot.f data.Data.xt in
  let loss = Loss.compute jdot.loss_fn yt pred in
  Tensor.to_float0_exn loss

let fit jdot data =
  let batches = Data.create_batches data jdot.config.Config.batch_size in
  let num_batches = List.length batches in
  
  let best_loss = ref Float.max_float in
  let patience = jdot.config.Config.early_stopping_patience in
  let mutable patience_counter = 0 in
  
  try
    for i = 1 to jdot.config.Config.num_iterations do
      let batch_idx = i mod num_batches in
      let (xs_batch, ys_batch, xt_batch) = List.nth batches batch_idx in
      
      let loss = train_step jdot xs_batch ys_batch xt_batch in
      
      if i mod 100 = 0 then begin
        let eval_loss = evaluate jdot data in
        Printf.printf "Iteration %d, Training Loss: %f, Evaluation Loss: %f\n" i loss eval_loss;
        
        if eval_loss < !best_loss then begin
          best_loss := eval_loss;
          patience_counter <- 0;
        end else begin
          patience_counter <- patience_counter + 1;
          if patience_counter >= patience then
            raise (Failure "Early stopping")
        end
      end
    done
  with
  | Failure msg -> Printf.printf "Training stopped: %s\n" msg
  | exn -> Printf.printf "An error occurred during training: %s\n" (Printexc.to_string exn)

let predict jdot x =
  jdot.f x

let save jdot filename =
  Serialize.save_multi ~filename [
    "model", Var_store.all_vars jdot.optimizer;
    "config", Serialize.of_string (to_string (Config.to_json jdot.config));
    "loss_fn", Serialize.of_string (Loss.to_string jdot.loss_fn);
  ]

let load filename =
  let vars = Serialize.load_multi ~filename in
  let config_json = Serialize.to_string (List.assoc "config" vars) in
  let config = Config.of_json (from_string config_json) in
  let loss_fn = Loss.of_string (Serialize.to_string (List.assoc "loss_fn" vars)) in
  let model_type = Model.of_json (from_string config_json) in
  let jdot = create model_type loss_fn config in
  Var_store.load ~filename:filename ~name:"model" (Var_store.all_vars jdot.optimizer);
  jdot