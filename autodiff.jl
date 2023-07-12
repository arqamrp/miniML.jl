# we go from matrix 1 (mxn) and 2 (nxp) to matrix 3 (mxp) through matmult
# grad3 is mxp
# grad1->3 is mxp x mxn
# grad1 is mxn: elementwise mult of grad3

function backward!(x::Tensor)

    #UNARY OPS
    # exp
    if x._op == "exp"
        if x._prev[1]._flag
            x._prev[1].grad += x.data
        end

    # log
    elseif x._op == "log"
        if x._prev[1]._flag
            x._prev[1].grad += 1 ./ (x._prev[1].data)
        end
    # transpose
    elseif x._op == "T"
        if x._prev[1]._flag
            x._prev[1].grad += x.grad'
        end
    
    elseif x._op == "getindex"
        idx = x._prev[2]
        if x._prev[1]._flag
            x._prev[1].grad[idx...] += x.grad
        end
        
    # reshaping and stuff
    
    elseif x._op == "reshape"
        if x._prev[1]._flag
            x._prev[1].grad += reshape(x.grad, x._prev[1].shape)
        end
    
#     elseif x._op
    
    # MIXED TENSOR-SCALAR OPS
    # ^
    elseif x._op == "^"
        if x._prev[1]._flag
            k = x._prev[2]
            x._prev[1].grad += (k) * (x._prev[1].data .^ (k-1))
        end
    
    # Scalar Tensor addition
    elseif x._op== "scalar+"
        if x._prev[1]._flag
            x._prev[1].grad += x.grad
        end
        
    # Scalar tensor multiplication
    elseif x._op== "scalar*"
        if x._prev[1]._flag
            x._prev[1].grad += x.grad * x._prev[2]
        end
        
    # Scalar division (k/tensor)
    elseif x._op == "/"
        if x._prev[1]._flag
            x._prev[1].grad += x._prev2 ./ (x._prev[1].data .^ 2)
        end
        
    # Scalar comparisons are assumed to have zero gradients
    
    #BINARY OPS
    # Matrix multiplication
    elseif x._op == "*"
        if x._prev[1]._flag
            x._prev[1].grad += x.grad * x._prev[2].data' # += since
        end
        if x._prev[2]._flag
            x._prev[2].grad +=  x._prev[1].data' * x.grad
        end
        
    # Tensor addition
    elseif x._op == "+"
        if x._prev[1]._flag
            x._prev[1].grad += x.grad
        end
        if x._prev[2]._flag
            x._prev[2].grad += x.grad
        end
    
    # Elementwise multiplication
    elseif x._op == "elemwise"
        if x._prev[1]._flag
            x._prev[1].grad += x._prev[2].data
        end
        if x._prev[2]._flag
            x._prev[2].grad += x._prev[1].data
        end
        
    end

end