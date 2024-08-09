let save_results filename data =
  let oc = open_out filename in
  List.iter (fun (x, y1, y2, y3) ->
    Printf.fprintf oc "%f,%f,%f,%f\n" x y1 y2 y3
  ) data;
  close_out oc

let plot_results filename xlabel ylabel title =
  let gnuplot = Unix.open_process_out "gnuplot" in
  Printf.fprintf gnuplot "set terminal png\n";
  Printf.fprintf gnuplot "set output '%s.png'\n" filename;
  Printf.fprintf gnuplot "set xlabel '%s'\n" xlabel;
  Printf.fprintf gnuplot "set ylabel '%s'\n" ylabel;
  Printf.fprintf gnuplot "set title '%s'\n" title;
  Printf.fprintf gnuplot "plot '%s' using 1:2 with lines title 'Theoretical', \
                               '' using 1:3 with lines title 'Empirical Mean', \
                               '' using 1:3:4 with yerrorbars title 'Empirical Std'\n" filename;
  close_out gnuplot