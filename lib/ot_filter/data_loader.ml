open Torch

let create_dataloader (features, labels) ~batch_size =
  let dataset = Dataset.of_tensors [features; labels] in
  Dataset.to_dataloaders [dataset] ~batch_size ~shuffle:true

let load_cifar10 ~path =
  let (train_images, train_labels), (test_images, test_labels) =
    Cifar.read_files ~path
  in
  let train_images = Tensor.reshape train_images [-1; 3; 32; 32] in
  let test_images = Tensor.reshape test_images [-1; 3; 32; 32] in
  (train_images, train_labels), (test_images, test_labels)

let load_cifar100 ~path =
  let (train_images, train_labels), (test_images, test_labels) =
    Cifar.read_files ~path ~num_classes:100
  in
  let train_images = Tensor.reshape train_images [-1; 3; 32; 32] in
  let test_images = Tensor.reshape test_images [-1; 3; 32; 32] in
  (train_images, train_labels), (test_images, test_labels)

let load_clothing1m ~path =
  let read_image_folder folder =
    let files = Sys.readdir (Filename.concat path folder) in
    let images = Array.map (fun file ->
      let img = Image.load (Filename.concat (Filename.concat path folder) file) in
      let tensor = Image.to_tensor img |> Tensor.reshape [-1; 3; 224; 224] in
      Tensor.div_scalar tensor (Tensor.of_float 255.)
    ) files in
    Tensor.stack (Array.to_list images) 0
  in
  let read_label_file file =
    let ic = open_in file in
    let labels = ref [] in
    (try
      while true do
        labels := int_of_string (input_line ic) :: !labels
      done
    with End_of_file -> close_in ic);
    Tensor.of_int1 (Array.of_list (List.rev !labels))
  in
  let train_images = read_image_folder "train" in
  let train_labels = read_label_file (Filename.concat path "train_labels.txt") in
  let test_images = read_image_folder "test" in
  let test_labels = read_label_file (Filename.concat path "test_labels.txt") in
  (train_images, train_labels), (test_images, test_labels)

let load_animal10n ~path =
  let read_image_folder folder =
    let files = Sys.readdir (Filename.concat path folder) in
    let images = Array.map (fun file ->
      let img = Image.load (Filename.concat (Filename.concat path folder) file) in
      let tensor = Image.to_tensor img |> Tensor.reshape [-1; 3; 64; 64] in
      Tensor.div_scalar tensor (Tensor.of_float 255.)
    ) files in
    Tensor.stack (Array.to_list images) 0
  in
  let read_label_file file =
    let ic = open_in file in
    let labels = ref [] in
    (try
      while true do
        labels := int_of_string (input_line ic) :: !labels
      done
    with End_of_file -> close_in ic);
    Tensor.of_int1 (Array.of_list (List.rev !labels))
  in
  let train_images = read_image_folder "train" in
  let train_labels = read_label_file (Filename.concat path "train_labels.txt") in
  let test_images = read_image_folder "test" in
  let test_labels = read_label_file (Filename.concat path "test_labels.txt") in
  (train_images, train_labels), (test_images, test_labels)