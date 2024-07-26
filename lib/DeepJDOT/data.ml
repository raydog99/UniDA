open Torch

module Data = struct
  type dataset = {
    images: Tensor.t;
    labels: Tensor.t;
  }

  let load_mnist () =
    try
      let mnist = Mnist.read_files ~train_image:"train-images-idx3-ubyte" 
                                   ~train_label:"train-labels-idx1-ubyte"
                                   ~test_image:"t10k-images-idx3-ubyte"
                                   ~test_label:"t10k-labels-idx1-ubyte" in
      let train_images = Tensor.reshape mnist.train_images ~shape:[-1; 1; 28; 28] in
      let train_labels = Tensor.of_int1 mnist.train_labels in
      { images = train_images; labels = train_labels }
    with
    | exn -> 
        Printf.eprintf "Error loading MNIST dataset: %s\n" (Printexc.to_string exn);
        exit 1

  let create_dataloader dataset batch_size shuffle =
    let dataset_length = Tensor.shape dataset.images |> List.hd in
    let indices = Tensor.arange ~start:0 ~end_:(Float.of_int dataset_length) ~options:(T Int64, Cpu) in
    let indices = if shuffle then Tensor.randperm indices else indices in
    
    let get_batch index =
      let start_idx = index * batch_size in
      let end_idx = min (start_idx + batch_size) dataset_length in
      let batch_indices = Tensor.narrow indices ~dim:0 ~start:start_idx ~length:(end_idx - start_idx) in
      let batch_images = Tensor.index_select dataset.images ~dim:0 ~index:batch_indices in
      let batch_labels = Tensor.index_select dataset.labels ~dim:0 ~index:batch_indices in
      (batch_images, batch_labels)
    in
    
    let num_batches = (dataset_length + batch_size - 1) / batch_size in
    (num_batches, get_batch)
end