### Question 1
(a) Based on the plot, the data is linearly separable, as we can draw a line between 1 and 5,6. Here, the distance between the line and 1 and the line and 5 or 6 will be the same. This is the line that maximizes the margin. Because these points 1,5,6 define the maximum margin linear boundary, they are the support vectors.

(b) We know that support vectors must satisfy $y_{i}(w^{T}x_{i} + b) = 1$.

For point 1, it must be $w^{T}(1,1) + b = 1$, which implies $w_{1} + w_{2} + b = 1$.

For point 5, it must be $w^{T}(1, 0) + b = -1$, which implies $w_{1} + b = -1$.

For point 6, it must be $w^{T}(0, 1) + b = -1$, which implies $w_{2} + b = -1$.

Thus we know that $w_{1} = -1 - b$ and $w_{2} = -1 - b = w_{1}$, so $w_{1} + w_{2} + b = -2 - 2b + b = 1$, which implies $b = -3$, then $w_{1} - 3 = -1$ so $w_{1} = w_{2} = 2$.

Hence we have $w_{1}x_{1} + w_{2}x_{2} + b = 2x_{1} + 2x_{2} - 3 = 0$.

Then, we know the geometric margin (distance from a support vector to the max margin separating line) is $1/||w|| = 1/\sqrt{4 + 4} = 1/\sqrt{8} \approx 0.3536$.

### Question 2
(a) We resolve distance ties by always taking the assignment from the lowest index cluster, e.g., if clusters 0 and 1 have the same lowest distance to an instance, we assign that instance to cluster 0. If a cluster is empty, we retain its previous centroid.

(b) Visually, it's easy to see the four separate generating groups. When $k=2$, we see that one of the clusters that was found was one of those exact generating clusters. However, the other 3 generating clusters were awkwardly clustered together, with the centroid in the middle of them, not representing them well. When $k=3$, two of the generating clusters are labeled correctly, then there is a centroid in the middle of the other two generating clusters, again splitting them. When $k=4$, all four generating clusters are recovered; this is exactly what we want it to look like. When $k=6$ and $k=10$, there exist some clusters that align with the generating clusters, but some of the generating clusters are themselves split into multiple clusters, from 3 to 5 depending on $k$. So we can see that when $k \neq 4$, we are always either taking multiple generating clusters and grouping them together (if $k<4$) or splitting a given generating cluster into more clusters (if $k>4$).

### Question 3
(a) The mean accuracy on the clean test set is 0.867.

(b) The mean accuracy using the noisy labels is approximately 0.828, and the mean accuracy using the repaired labels is 0.8644. The difference between repaired and noisy accuracy means is approximately 0.03645. Here, we see that K-NN repair helped increase accuracy by over 3.5pts. The reason why this K-NN helped is because the seed data of the positive class is likely closer to other instances of seed data of the positive class (in many cases) than it is to instances of the negative class. We see this assumption seems to hold in this case, as the repair is able to fix some of the flips. However, this is a property of the way the input data were generated; it is possible to have data from the same class that is not close in euclidean distance, but we still want to classify together. Based on such properties, it is possible that doing the 'repair' could actually hurt the accuracy even more, instead of recovering some lost accuracy from the noising process.