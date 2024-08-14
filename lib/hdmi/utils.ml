open Torch

let generate_dummy_data input_channels num_classes num_samples batch_size =
  List.init num_samples (fun _ -> 
    (Tensor.rand [batch_size; input_channels; 224; 224], Tensor.randint ~high:num_classes [batch_size])
  )

let generate_dummy_target_data input_channels num_samples batch_size =
  List.init num_samples (fun _ -> Tensor.rand [batch_size; input_channels; 224; 224])