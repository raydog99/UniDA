open Torch
open Base
open Data
open Icnn

type t = {
	primal_network: Icnn.t;
	dual_network: Icnn.t;
	optimizer: Optimizer.t;
	lambda: float;
}

let create input_dim hidden_dims output_dim learning_rate lambda =
	let primal_net = Icnn.create input_dim hidden_dims output_dim in
	let dual_net = Icnn.create output_dim (List.rev hidden_dims) input_dim in
	let parameters = Icnn.parameters primal_net @ Icnn.parameters dual_net in
	let optimizer = Optimizer.adam parameters ~lr:learning_rate in
	{ primal_network = primal_net;
	  dual_network = dual_net;
	  optimizer = optimizer;
	  lambda = lambda }

let forward_map gan x =
	Icnn.gradient gan.primal_network x

let inverse_map gan y =
	Icnn.gradient gan.dual_network y

let correlation_loss gan x y =
	let primal_val = Icnn.forward gan.primal_network x in
	let dual_val = Icnn.forward gan.dual_network y in
	let inner_product = Tensor.(mean (x * (forward_map gan y))) in
	Tensor.(mean (primal_val + inner_product - dual_val))

let cycle_consistency_loss gan y =
	let inverse_y = inverse_map gan y in
	let forward_inverse_y = forward_map gan inverse_y in
	Tensor.(mean (pow (y - forward_inverse_y) (Scalar.f 2.)))

let total_loss gan x y =
	let corr_loss = correlation_loss gan x y in
	let cycle_loss = cycle_consistency_loss gan y in
	Tensor.(corr_loss + (f gan.lambda * cycle_loss))

let train_step gan x y =
	let loss = total_loss gan x y in
		Optimizer.zero_grad gan.optimizer;
		Tensor.backward loss;
		Optimizer.step gan.optimizer;
	loss

let train gan ~num_epochs ~batch_size ~data_loader ~validation_loader =
	let rec train_epoch epoch best_val_loss =
	  if epoch > num_epochs then
	    gan
	  else
	    let train_loss = ref 0. in
	    let num_batches = ref 0 in
	    Dataset.iter data_loader ~f:(fun (x, y) ->
	      let loss = train_step gan x y in
	      train_loss := !train_loss +. Tensor.float_value loss;
	      num_batches := !num_batches + 1
	    );
	    let avg_train_loss = !train_loss /. Float.of_int !num_batches in
	    
	    let val_loss = ref 0. in
	    let num_val_batches = ref 0 in
	    Dataset.iter validation_loader ~f:(fun (x, y) ->
	      let loss = total_loss gan x y in
	      val_loss := !val_loss +. Tensor.float_value loss;
	      num_val_batches := !num_val_batches + 1
	    );
	    let avg_val_loss = !val_loss /. Float.of_int !num_val_batches in
	    
	    Stdio.printf "Epoch %d, Train Loss: %f, Validation Loss: %f\n" epoch avg_train_loss avg_val_loss;
	    
	    if Float.(avg_val_loss < best_val_loss) then (
	      save gan (Printf.sprintf "w2gan_model_epoch_%d.ot" epoch);
	      train_epoch (epoch + 1) avg_val_loss
	    ) else
	      train_epoch (epoch + 1) best_val_loss
	in
	train_epoch 1 Float.infinity

let sample gan num_samples =
	let x = Tensor.randn [num_samples; Icnn.input_dim gan.primal_network] in
	forward_map gan x

let save gan filename =
	let state_dict = [
	  ("primal_network", Icnn.state_dict gan.primal_network);
	  ("dual_network", Icnn.state_dict gan.dual_network);
	  ("lambda", Tensor.f gan.lambda);
	] in
	Serialize.save ~filename state_dict

let load filename learning_rate =
	let state_dict = Serialize.load ~filename in
	let primal_net = Icnn.load (List.Assoc.find_exn state_dict ~equal:String.equal "primal_network") in
	let dual_net = Icnn.load (List.Assoc.find_exn state_dict ~equal:String.equal "dual_network") in
	let lambda = Tensor.float_value (List.Assoc.find_exn state_dict ~equal:String.equal "lambda") in
	let input_dim = Icnn.input_dim primal_net in
	let output_dim = Icnn.output_dim primal_net in
	create input_dim [] output_dim learning_rate lambda