module T = Torch.Tensor

type error =
  | InvalidDimensions of string
  | InvalidParameter of string

exception ClusteringError of error

let check_dimensions tensor expected_dims error_msg =
  let actual_dims = T.shape tensor in
  if actual_dims <> expected_dims then
    raise (ClusteringError (InvalidDimensions error_msg))

let check_positive_float param name =
  if param <= 0. then
    raise (ClusteringError (InvalidParameter (Printf.sprintf "%s must be positive" name)))

let rec last = function
  | [] -> failwith "Empty list"
  | [x] -> x
  | _ :: t -> last t

let string_of_list to_string lst =
  "[" ^ (String.concat "; " (List.map to_string lst)) ^ "]"