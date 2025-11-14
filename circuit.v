module mux2to1(
  input A, B, Sel
  output Y
);
wire nSel, w1, w2;
not (nSel, Sel);
and (w1, A, nSel);
and (w2, B, Sel);
or  (Y, w1, w2);
endmodule
