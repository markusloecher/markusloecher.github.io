### Random Walks

Suppose we start with *n* dollars, and make a sequence of bets. For each
bet, we win 1 dollar with probability *p*, and lose 1 dollar with
probability 1 − *p*. We quit if either we go broke, in which case we
lose, or when we reach *T* = *n* + *m* dollars, that is, when we win *m*
dollars. For example, in Roulette, *p* = 18/38 = 9/19 ∼ 0.473. If
*n* = 100 dollars, and *m* = 100 dollars, then *T* = 200 dollars. What
are the odds we win 100 dollars before losing 100 dollars? Most folks
would think that since 0.473 ∼ 0.5, the odds are not so bad. In fact, as
we will see, we win before we lose with probability at most 1/37649!

### Boundary Crossing

This random walk is a special type of random walk where moves are
independent of the past, and is called a . If *p* = 1/2, the random walk
is unbiased, whereas if *p* ≠ 1/2, the random walk is biased. If the
walk hits a boundary (0 or *n* + *m*), then we stop playing.

<iframe width="1000" height="900" scrolling="no" frameborder="no" src="https://nbwr.shinyapps.io/gamblers-ruin/">
</iframe>

### Probability to win

Let’s figure out the probability that we gain m before losing n.~Let
*T* = *n* + *m* and *D*<sub>*t*</sub> denote the number of dollars we
have at time step *t*. Let
*P*<sub>*n*</sub> = *P**r*(*D*<sub>*t*</sub> ≥ *T*|*D*<sub>0</sub> = *n*)
be the probability we get *T* before we go broke, given that we start
with *n* dollars. Our question then, is what is *P*<sub>*n*</sub> ?
We’re going to use a recursive approach.
*P*<sub>*n*</sub> = *p* ⋅ *P*<sub>*n* + 1</sub> + (1 − *p*) ⋅ *P*<sub>*n* − 1</sub>,  for 0 &lt; *n* &lt; *T*
Solving it for *p* &lt; 0.5 yields

$$
P\_n = \\frac{\\left( \\frac{1-p}{p} \\right)^n - 1}{\\left( \\frac{1-p}{p} \\right)^T - 1} \\leq \\left( \\frac{p}{1-p} \\right)^m 
$$
This last expression is even independent of n! It is also exponentially
small in m!
