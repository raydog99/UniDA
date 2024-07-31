open Base
open Torch

let load_cifar10_lt imbalance_ratio num_labeled num_unlabeled =
  let num_classes = 10 in
  let input_dim = 3 * 32 * 32 in
  
  let create_random_data count =
    List.init count ~f:(fun _ ->
      let x = Tensor.rand [32; 32; 3] in
      let y = Tensor.of_int0 (Random.int num_classes) in
      (x, y)
    )
  in
  
  let labeled_data = create_random_data num_labeled in
  let unlabeled_data = List.map (create_random_data num_unlabeled) ~f:fst in
  
  (labeled_data, unlabeled_data)

let create_batches data batch_size =
  List.groupi data ~break:(fun i _ _ -> i % batch_size = 0)