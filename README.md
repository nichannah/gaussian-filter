
Gaussing filter/blur in Fortran and Python. Edges are treated using reflection. Input can be masked. This operation is very close to conservative. 

Masking is handled in the following way: 

1) Masked points are skipped in the convolution, their value will be unchanged. 
2) Input points which are masked have weight 0 in the kernel. i.e. the kernel is effectively masked.  
3) Since the kernel should still have a sum of 1, the value of the masked part of the kernel is evenly distributed over the non-masked part of the kernel. So, for example, imagine a single point which is completely surrounded my masked points. The filter with have no effect at all on this point because the kernel will have a weight of one (1) surrounded by zeros.


