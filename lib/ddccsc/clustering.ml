open Torch
open Utils

type params = {
  n_clusters: int;
  epsilon: float;
  n_iter: int;
  learning_rate: float;
  n_epochs: int;
}

let init_centers n_clusters embedding_dim =
  check_positive_float (float_of_int n_clusters) "n_clusters";
  check_positive_float (float_of_int embedding_dim) "embedding_dim";
  Tensor.randn [n_clusters; embedding_dim]

let cluster data params =
  let n_samples, input_dim = Tensor.shape data in
  
  check_positive_float params.epsilon "epsilon";
  check_positive_float (float_of_int params.n_iter) "n_iter";
  check_positive_float params.learning_rate "learning_rate";
  check_positive_float (float_of_int params.n_epochs) "n_epochs";
  
  let ae = Autoencoder.create_autoencoder input_dim [500; 250; 10] in
  
  let centers = init_centers params.n_clusters 10 in
  
  let ae_optimizer = Optimizer.Adam.create ~learning_rate:params.learning_rate [] in
  let centers_optimizer = Optimizer.Adam.create ~learning_rate:params.learning_rate [] in
  
  for epoch = 1 to params.n_epochs do
    let embeddings, reconstructed = Autoencoder.forward ae data in
    
    let rec_loss = Autoencoder.reconstruction_loss data reconstructed in
    let ot_loss = Optimal_transport.ot_loss embeddings centers params.epsilon params.n_iter in
    let total_loss = Tensor.add rec_loss ot_loss in
    
    Tensor.backward total_loss;
    Optimizer.Adam.step ae_optimizer;
    Optimizer.Adam.step centers_optimizer;
    
    Optimizer.Adam.zero_grad ae_optimizer;
    Optimizer.Adam.zero_grad centers_optimizer;
    
    if epoch mod 10 = 0 then
      Printf.printf "Epoch %d: Loss = %f\n" epoch (Tensor.float_value total_loss)
  done;
  
  (ae, centers)

let assign_clusters ae centers data =
  let embeddings = ae.Autoencoder.encoder data in
  let distances = Tensor.cdist embeddings centers in
  Tensor.argmin distances ~dim:1 ~keepdim:false