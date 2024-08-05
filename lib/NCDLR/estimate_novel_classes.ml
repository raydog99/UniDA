open Torch
open Types

let kmeans data k max_iter =
  let n = Tensor.shape data |> List.hd in
  let d = Tensor.shape data |> List.nth 1 in
  let centroids = Tensor.narrow data ~dim:0 ~start:0 ~length:k in
  
  let rec loop iter centroids =
    if iter >= max_iter then centroids
    else
      let distances = Tensor.cdist data centroids in
      let assignments = Tensor.argmin distances ~dim:1 ~keepdim:false in
      let new_centroids = Tensor.zeros [k; d] in
      let counts = Tensor.zeros [k] in
      
      for i = 0 to n - 1 do
        let cluster = Tensor.get assignments [i] |> Tensor.item in
        Tensor.(set new_centroids [cluster] (add (get new_centroids [cluster]) (get data [i])));
        Tensor.(set counts [cluster] (add (get counts [cluster]) (Scalar.float 1.)))
      done;
      
      let new_centroids = Tensor.div new_centroids (Tensor.reshape counts [k; 1]) in
      if Tensor.equal centroids new_centroids then new_centroids
      else loop (iter + 1) new_centroids
  in
  
  let final_centroids = loop 0 centroids in
  let distances = Tensor.cdist data final_centroids in
  Tensor.argmin distances ~dim:1 ~keepdim:false

let cluster_data data num_clusters =
  kmeans data num_clusters 100

let evaluate_clustering clusters known_labels num_known_classes =
  let n = Tensor.shape clusters |> List.hd in
  let confusion_matrix = Tensor.zeros [num_known_classes; num_known_classes] in
  
  for i = 0 to n - 1 do
    let true_label = Tensor.get known_labels [i] |> Tensor.item |> int_of_float in
    let pred_label = Tensor.get clusters [i] |> Tensor.item |> int_of_float in
    if true_label < num_known_classes && pred_label < num_known_classes then
      Tensor.(set confusion_matrix [true_label; pred_label] 
        (add (get confusion_matrix [true_label; pred_label]) (Scalar.float 1.)))
  done;
  
  let acc_s = Tensor.(sum (diag confusion_matrix) / (sum confusion_matrix)) |> Tensor.to_float0_exn in
  let acc_c = Tensor.(mean (div (diag confusion_matrix) (sum confusion_matrix ~dim:1))) |> Tensor.to_float0_exn in
  
  acc_s, acc_c

let estimate_novel_classes known_data novel_data max_k_u beta =
  let combined_data = Tensor.cat [known_data.images; novel_data.images] ~dim:0 in
  let known_labels = known_data.labels in
  let num_known_classes = Tensor.(max known_labels).to_int0_exn + 1 in

  let rec binary_search low high =
    if low >= high then low 
    else
      let mid = (low + high) / 2 in
      let k_u = mid - num_known_classes in
      let clusters = cluster_data combined_data (num_known_classes + k_u) in
      let acc_s, acc_c = evaluate_clustering clusters known_labels num_known_classes in
      let acc = beta *. acc_s +. (1. -. beta) *. acc_c in
      if acc > 0.5 then binary_search mid high
      else binary_search low (mid - 1)
  in

  binary_search num_known_classes max_k_u