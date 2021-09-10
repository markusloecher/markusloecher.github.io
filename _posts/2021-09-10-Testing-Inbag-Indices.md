## Are the inbag indices consistent ?

I believe that the only stochastic elements of a random forest are the
row and column subsampling. For the second forest/tree I disable the
former by setting various parameters and the latter as well by choosing
`mtry = p`. So I would expect the two forests (consisting of a single
tree) to be identical ?

I see deviations from this for `ranger` and `randomForest` but `cforest`
gives indeed identical predictions.

### ranger

    set.seed(123)

    rf1 = ranger(medv ~ ., data = Boston, mtry = p, keep.inbag = TRUE, num.trees = 1)
    #seems to work: rep(1:5, c(1,0,2,1,0))
    inbag_indices = rep(1:n, rf1$inbag.counts[[1]])
    oob = rf1$inbag.counts[[1]] == 1
    OOB = as.numeric(oob)

    set.seed(123)#should make no difference ?
    rf2 = ranger(medv ~ ., data = Boston[inbag_indices,], mtry = p, keep.inbag = TRUE, num.trees = 1, replace = FALSE, sample.fraction=1)

    #check: 
    all(rf2$inbag.counts[[1]] == 1)

    ## [1] TRUE

    p1 = predict(rf1, Boston)
    p2 = predict(rf2, Boston)

    plot(p1$predictions, p2$predictions, pch=20, cex = 0.75, col= rgb(1-OOB,0,OOB,0.5), xlab = "rf with bootstrap", ylab = "rf with inbag", main = "ranger");grid()
    abline(a=0,b=1,col=2)

![](/assets/inbag-indices/unnamed-chunk-4-1.png)

### randomForest

    library(randomForest)
    set.seed(123)

    rf1 = randomForest(medv ~ ., data = Boston, mtry = p, keep.inbag = TRUE, ntree = 1)
    #seems to work: rep(1:5, c(1,0,2,1,0))
    inbag_indices = rep(1:n, rf1$inbag[,1])
    oob = rf1$inbag[,1] == 1
    OOB = as.numeric(oob)

    set.seed(123)#should make no difference ?
    rf2 = randomForest(medv ~ ., data = Boston[inbag_indices,], mtry = p, keep.inbag = TRUE, ntree = 1, replace = FALSE, sampsize=n)

    #check: 
    all(rf2$inbag == 1)

    ## [1] TRUE

    p1 = predict(rf1, Boston)
    p2 = predict(rf2, Boston)

    plot(p1, p2, pch=20, cex = 0.75, col= rgb(1-OOB,0,OOB,0.5), xlab = "rf with bootstrap", ylab = "rf with inbag", main = "randomForest");grid()
    abline(a=0,b=1,col=2)

![](/assets/inbag-indices/unnamed-chunk-6-1.png)

### cforest

    library(partykit)

    set.seed(123)

    cf1 = cforest(medv ~ ., data = Boston, mtry = p, ntree = 1, perturb = list(replace = TRUE))
    #seems to work: rep(1:5, c(1,0,2,1,0))
    inbag_indices = rep(1:n, weights(cf1)[[1]])
    oob = weights(cf1)[[1]] == 1
    OOB = as.numeric(oob)

    set.seed(123)#should make no difference ?
    cf2 = cforest(medv ~ ., data = Boston[inbag_indices,], mtry = p,  ntree = 1, perturb = list(replace = FALSE, fraction=1))

    #check: 
    all(weights(cf2)[[1]] == 1)

    ## [1] TRUE

    p1 = predict(cf1, Boston)
    p2 = predict(cf2, Boston)

    plot(p1, p2, pch=20, cex = 0.75, col= rgb(1-OOB,0,OOB,0.5), xlab = "rf with bootstrap", ylab = "rf with inbag", main = "cforest");grid()
    abline(a=0,b=1,col=2)

![](/assets/inbag-indices/unnamed-chunk-8-1.png)
