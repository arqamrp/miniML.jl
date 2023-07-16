# LOSS FUNCTIONS

# Sum of squared errors
function SSE(y_true::Tensor, y_pred::Tensor)
    diff2 = (y_true - y_pred)^2
    sse = sum(diff2, 1)
    return sse
end


function MSE(y_true::Tensor, y_pred::Tensor)
    diff2 = (y_true - y_pred)^2
    mse = sum(diff2, 1)
end

# Binary cross entropy
function binary_cross_entropy(logits::Tensor, ground_truth::Tensor)
    
end
