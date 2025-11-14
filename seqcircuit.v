// Assume DFF module exists 
module seq_ckt ( 
input A, clk
output Q 
); 
wire w1, w2, w3; 
dff ff1 (w3, A, clk);
dff ff1 (w2, w1, clk);
and (w1, A, w2);      
or (Q, w2, w3)  
endmodule
