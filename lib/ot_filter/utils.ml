open Torch

let pairwise_distances x y =
  let x2 = Tensor.sum (Tensor.pow x 2.) 1 |> Tensor.unsqueeze 1 in
  let y2 = Tensor.sum (Tensor.pow y 2.) 1 |> Tensor.unsqueeze 0 in
  let xy = Tensor.mm x (Tensor.transpose y 0 1) in
  Tensor.add (Tensor.add x2 y2) (Tensor.mul xy (Tensor.of_float (-2.)))

let sinkhorn_knopp a b k epsilon max_iter =
  let k = Tensor.div k (Tensor.of_float epsilon) in
  let u = Tensor.ones_like a in
  let v = Tensor.ones_like b in
  let rec iterate i u v =
    if i >= max_iter then (u, v)
    else
      let u_new = Tensor.div a (Tensor.mv k v) in
      let v_new = Tensor.div b (Tensor.mv (Tensor.transpose k 0 1) u_new) in
      iterate (i + 1) u_new v_new
  in
  let u, v = iterate 0 u v in
  Tensor.mul (Tensor.mul (Tensor.diag u) k) (Tensor.diag v)

let sharpen probs temperature =
  let powered_probs = Tensor.pow probs (Tensor.of_float (1. /. temperature)) in
  Tensor.div powered_probs (Tensor.sum powered_probs 1 |> Tensor.unsqueeze 1)

let mixup x_a x_b y_a y_b alpha =
  let batch_size = Tensor.shape x_a |> List.hd in
  let lambda = Tensor.rand [batch_size] ~low:0. ~high:1. |> Tensor.unsqueeze 1 in
  let mixed_x = Tensor.add (Tensor.mul x_a lambda) (Tensor.mul x_b (Tensor.sub (Tensor.of_float 1.) lambda)) in
  let mixed_y = Tensor.add (Tensor.mul y_a lambda) (Tensor.mul y_b (Tensor.sub (Tensor.of_float 1.) lambda)) in
  mixed_x, mixed_y

let add_synthetic_noise labels noise_rate symmetric =
  let num_classes = Tensor.max labels |> Tensor.item |> int_of_float |> (+) 1 in
  let noisy_labels = Tensor.clone labels in
  Tensor.iteri_all labels (fun idx label ->
    if Tensor.rand [] |> Tensor.item < noise_rate then
      let new_label = 
        if symmetric then
          Tensor.randint ~to_:num_classes [1] |> Tensor.item |> int_of_float
        else
          (int_of_float label + 1) mod num_classes
      in
      Tensor.set noisy_labels idx (Tensor.of_int new_label)
  );
  noisy_labels

let accuracy pred_labels true_labels =
  let correct = Tensor.sum (Tensor.eq pred_labels true_labels) in
  Tensor.float_value correct /. float_of_int (Tensor.shape true_labels |> List.hd)

let augment_image image =
  let batch_size, c, h, w = Tensor.shape4_exn image in
  let image = Tensor.detach image in
  (* Random crop *)
  let padded = Tensor.pad image [4; 4; 4; 4] in
  let crop_h = Random.int 9 in
  let crop_w = Random.int 9 in
  let cropped = Tensor.narrow padded ~dim:2 ~start:crop_h ~length:32
              |> Tensor.narrow ~dim:3 ~start:crop_w ~length:32 in
  (* Random horizontal flip *)
  let flipped = 
    if Random.bool () then Tensor.flip cropped ~dims:[3]
    else cropped
  in
  flipped

let augment_batch batch =
  Tensor.map batch ~f:augment_image