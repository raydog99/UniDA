open Torch

type block = {
  conv1: Layer.t;
  bn1: Layer.t;
  conv2: Layer.t;
  bn2: Layer.t;
  shortcut: Layer.t option;
}

type t = {
  conv1: Layer.t;
  layer1: block list;
  layer2: block list;
  layer3: block list;
  layer4: block list;
  bn: Layer.t;
  fc: Layer.t;
}

let create_block in_planes planes stride =
  let conv1 = Layer.conv2d in_planes planes ~kernel_size:3 ~stride ~padding:1 ~bias:false in
  let bn1 = Layer.batch_norm2d planes in
  let conv2 = Layer.conv2d planes planes ~kernel_size:3 ~stride:1 ~padding:1 ~bias:false in
  let bn2 = Layer.batch_norm2d planes in
  let shortcut = 
    if stride <> 1 || in_planes <> planes then
      Some (Layer.conv2d in_planes planes ~kernel_size:1 ~stride ~bias:false)
    else
      None
  in
  { conv1; bn1; conv2; bn2; shortcut }

let create_layer block in_planes planes num_blocks stride =
  let strides = stride :: List.init (num_blocks - 1) (fun _ -> 1) in
  let blocks = List.mapi (fun i s -> 
    block (if i = 0 then in_planes else planes) planes s
  ) strides in
  blocks

let create_preact_resnet18 num_classes =
  let conv1 = Layer.conv2d 3 64 ~kernel_size:3 ~stride:1 ~padding:1 ~bias:false in
  let layer1 = create_layer create_block 64 64 2 1 in
  let layer2 = create_layer create_block 64 128 2 2 in
  let layer3 = create_layer create_block 128 256 2 2 in
  let layer4 = create_layer create_block 256 512 2 2 in
  let bn = Layer.batch_norm2d 512 in
  let fc = Layer.linear 512 num_classes in
  { conv1; layer1; layer2; layer3; layer4; bn; fc }

let forward_block x block =
  let out = Tensor.(x |> Layer.forward block.bn1 |> relu |> Layer.forward block.conv1) in
  let out = Tensor.(out |> Layer.forward block.bn2 |> relu |> Layer.forward block.conv2) in
  let shortcut = match block.shortcut with
    | Some layer -> Layer.forward layer x
    | None -> x
  in
  Tensor.(out + shortcut)

let forward model x =
  let out = Layer.forward model.conv1 x in
  let out = List.fold_left forward_block out model.layer1 in
  let out = List.fold_left forward_block out model.layer2 in
  let out = List.fold_left forward_block out model.layer3 in
  let out = List.fold_left forward_block out model.layer4 in
  let out = Tensor.(out |> Layer.forward model.bn |> relu) in
  let out = Tensor.adaptive_avg_pool2d out [1; 1] in
  let out = Tensor.reshape out [-1; 512] in
  Layer.forward model.fc out

let parameters model =
  let get_params layer = Layer.parameters layer in
  List.concat [
    get_params model.conv1;
    List.concat_map (fun block -> 
      List.concat [
        get_params block.conv1;
        get_params block.bn1;
        get_params block.conv2;
        get_params block.bn2;
        match block.shortcut with Some l -> get_params l | None -> []
      ]
    ) (model.layer1 @ model.layer2 @ model.layer3 @ model.layer4);
    get_params model.bn;
    get_params model.fc;
  ]