open Torch

type t = {
  mutable data: Tensor.t;
  mutable index: int;
  capacity: int;
}

let create capacity dim =
  { data = Tensor.zeros [capacity; dim]; index = 0; capacity }

let add buffer tensor =
  let tensor = Tensor.reshape tensor [-1; Tensor.shape buffer.data |> List.nth 1] in
  let num_samples = Tensor.shape tensor |> List.hd in
  let remaining = buffer.capacity - buffer.index in
  if num_samples <= remaining then
    (Tensor.narrow buffer.data ~dim:0 ~start:buffer.index ~length:num_samples) <- tensor
  else (
    (Tensor.narrow buffer.data ~dim:0 ~start:buffer.index ~length:remaining) <- Tensor.narrow tensor ~dim:0 ~start:0 ~length:remaining;
    (Tensor.narrow buffer.data ~dim:0 ~start:0 ~length:(num_samples - remaining)) <- Tensor.narrow tensor ~dim:0 ~start:remaining ~length:(num_samples - remaining)
  );
  buffer.index <- (buffer.index + num_samples) mod buffer.capacity

let get buffer =
  buffer.data