```
 COLOYVAIN K-modularity, k-means, or spectral co-clustering

   Mx, My, R = coloyvain(W, k)
   Mx, My, R = coloyvain(W, k, objective)
   Mx, My, R = coloyvain(X, Y, k, objective, similarity)
   Mx, My, R = coloyvain(_, objective, similarity, Name=Value)

   Inputs:

       W: Bipartite network matrix of size p x q.

       X: Data matrix of size n x p, where
          n is the number of data points and
          p is the number of features.

       Y: Data matrix of size n x q, where
          n is the number of data points and
          q is the number of features.

       k: Number of modules (positive integer).

       objective: Clustering objective.
           See LOYVAIN for all options.

       similarity: Type of similarity.
           See LOYVAIN for all options.

       Name=Value Arguments.
           See LOYVAIN for all Name=Value arguments.

   Outputs:
       Mx: Vector of module assignments for X (length p).
       My: Vector of module assignments for Y (length q).
       R: Value of maximized objective.

   Methodological notes:
       Coloyvain simultaneously clusters X and Y via Loyvain
       co-clustering of the cross-similarity matrix.

   See also:
       LOYVAIN, CANONCOV.

```
