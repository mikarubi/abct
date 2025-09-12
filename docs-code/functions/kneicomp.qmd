```
 KNEICOMP Components of neighbor matrices

   V = kneicomp(W, k)
   V = kneicomp(X, k)
   V = kneicomp(_, k, weight)
   V = kneicomp(_, k, weight, Name=Value)

   Inputs:
       W: Network matrix of size n x n.
       OR
       X: Data matrix of size n x p, where
           n is the number of data points and
           p is the number of features.

       k: Number of components.

       weight: Type of components
           "weighted": Weighted components (default).
           "binary": Binary components.

       Name=Value Arguments:
           KNEIGHBOR: type, kappa, similarity, method
               (see KNEIGHBOR for details).
           LOYVAIN: All Name=Value arguments
               (binary components only, see LOYVAIN for details).

   Outputs:
       V: Component matrix (size n x k).

   Methodological notes:
       By default, weighted components are eigenvectors of
       common-neighbors matrices. In imaging neuroscience, these
       components are approximately equivalent to co-activity gradients
       (diffusion-map embeddings).
 
       Correspondingly, binary components are modules of common-neighbors 
       matrices, estimated using the Loyvain algorithm. They are
       equivalent to eigenvectors of common-neighbors matrices with binary
       constraints. The order of binary components will be arbitrary. 

   See also:
       KNEIGHBOR, LOYVAIN.

```
