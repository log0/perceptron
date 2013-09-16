# Perceptron Learning Algorithm implementation
# 
# This script demonstrates what could happen if the bias neuron/intercept term/constant
# is missing. The result is that the algorithm does not converge.
# 
# Reason: Without the intercept term, there are certain decision boundaries that the
# perceptron cannot learn. The perceptron might have learned the direction of the boundary,
# but must move the decision boundary left or right to properly learn. However, the
# lack of an intercept term forces it to pass through the origin, or you can understand
# it as it cannot move left or right to learn the right decision boundary.
#
# Note: You may have to run this script multiple times to see the failure to learn a decision
# boundary.

perceptron <- function(X, Y)
{
    converged <- F

    # Initialize weight vector to 0, with NO bias neuron/constant/intercept term
    W <- matrix(0, 1, 2)

    # Run for 10000 iterations
    for (i in 1:10000)
    {
        # Calculate h(x) with the weight vector W and the data input X
        h.X <- fn.sign(W %*% t(X))

        # Calculate the misclassified mask
        misclassified.mask <- h.X != Y
        
        # Check if all of the points are classified correctly
        if (sum(misclassified.mask) == 0)
        {
            # Yes! We are done.
            converged <- T
            break
        }
        else
        {
            # No! We have to update the weight vector now using any one of the misclassified input
            
            # Get the misclassified points out
            misclassified.points <- X[misclassified.mask, , drop = F]
            misclassified.points.Y <- Y[misclassified.mask]
            
            # Get one of them
            misclassified.point.index <- sample(dim(misclassified.points)[1], 1)
            misclassified.point <- misclassified.points[misclassified.point.index, , drop = F]
            misclassified.point.Y <- misclassified.points.Y[misclassified.point.index]

            # Now update the weight vector
            W <- W + misclassified.point.Y %*% misclassified.point
        }
        
        # repeat
    }
    
    if (converged)
    {
        cat('Converged! Iteration ', i, ' , with final weight : ', W, '\n')
    }
    else
    {
        cat('DID NOT CONVERGE!\n')
    }
    
    return(W)
}

####################################################################################
# Function to determine if a point lies on the left or right side of a line, which
# is the ideal/learned decision boundary here
####################################################################################
on.which.side <- function(line.separator, point)
{
    values <- (line.separator[2,1] - line.separator[1,1]) * (point[,2] - line.separator[1,2]) -
            (line.separator[2,2] - line.separator[1,2]) * (point[,1] - line.separator[1,1])
    return(fn.sign(values))
}

####################################################################################
# The SIGN function in the perceptron learning algorithm
####################################################################################
fn.sign <- function(values)
{
    return(ifelse(values > 0, 1, -1))
}

####################################################################################
# This function calculates the two ending points of the decision boundary given
# a weight vector. Not necessarily in practice, but just to help visualize for
# you to see.
####################################################################################
decision.boundary <- function(W)
{
    X1 <- c(-2.0, 2.0)
    X2 <- -(W[1] * X1) / W[2]
    return(matrix(cbind(X1, X2), 2, 2))
}

# Initialize a random plane to separate the -1, +1
line.separator <- matrix(runif(4, -1, 1), 2, 2)
line.separator[1,2] <- -2
line.separator[2,2] <- +2

# Initialize N random points, and Y
N <- 10
X <- matrix(runif(N*2, -1, 1), N, 2)
Y <- on.which.side(line.separator, X)

# Run perceptron algorithm
W <- perceptron(X, Y)

# Plot the points according to its actual class
plot(X, pch = Y, xlim = c(-1.5, 1.5), ylim = c(-1.5, 1.5))

# Plot ideal decision boundary
lines(line.separator, col="green")

# Plot learned decision boundary
lines(decision.boundary(W), col="red")

title(paste("Perceptron training against N = ", N))
legend(0, -1.0, c("Ideal decision boundary", "Learned decision boundary"), lty=c(1,1), lwd=c(2.5,2.5),col=c("green","red"))