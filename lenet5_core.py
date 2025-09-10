
# In this file, I am going to try and write my own tiny version of LeNet-5 using only basic python.
# I am going to try and avoid external libraries. I will just use python lists and loops.
# First of, I am going to write some tiny helper functions for math that I need.

import math
import json
import os
import random

# I am going to use tanh like the original paper did
def tanh(x):
    # I will probably clamp a bit to be safe
    if x < -15: 
        return -1.0
    if x > 15:
        return 1.0
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def dtanh(y):
    # derivative in terms of output y = tanh(x) is 1 - y^2
    return 1.0 - (y*y)

def softmax(vec):
    # I will do a simple softmax with subtracting max for stability
    m = vec[0]
    for v in vec:
        if v > m:
            m = v
    exps = []
    s = 0.0
    for v in vec:
        e = math.exp(v - m)
        exps.append(e)
        s += e
    out = []
    for e in exps:
        out.append(e / s)
    return out

def zeros(shape):
    # I will try to make lists of zeros for any shape like (a,b,c)
    if len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    if len(shape) == 2:
        out = []
        for _ in range(shape[0]):
            out.append([0.0 for _ in range(shape[1])])
        return out
    if len(shape) == 3:
        out = []
        for _ in range(shape[0]):
            out.append(zeros(shape[1:]))
        return out
    if len(shape) == 4:
        out = []
        for _ in range(shape[0]):
            out.append(zeros(shape[1:]))
        return out
    return None

def he_uniform(fan_in, fan_out):
    # I am going to just do a small uniform init (not exactly He, but simple)
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return lambda: random.uniform(-limit, limit)

# I want to make the conv layers and dense layers.
# I am going to keep this very simple. No padding inside conv (the input is already 32x32), stride=1.

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # weights shape: out_channels x in_channels x k x k
        # bias shape: out_channels
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        init = he_uniform(fan_in, fan_out)
        self.W = []
        for oc in range(out_channels):
            w_ic = []
            for ic in range(in_channels):
                kernel = []
                for kr in range(kernel_size):
                    row = []
                    for kc in range(kernel_size):
                        row.append(init())
                    kernel.append(row)
                w_ic.append(kernel)
            self.W.append(w_ic)
        self.b = [0.0 for _ in range(out_channels)]

        # slots for gradients
        self.dW = zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.db = [0.0 for _ in range(out_channels)]
        self.last_input = None

    def forward(self, x):
        # x shape: in_channels x H x W
        self.last_input = x
        H = len(x[0])
        W = len(x[0][0])
        k = self.kernel_size
        out_h = H - k + 1
        out_w = W - k + 1
        out = []
        for oc in range(self.out_channels):
            oc_map = []
            for r in range(out_h):
                row_vals = []
                for c in range(out_w):
                    s = 0.0
                    for ic in range(self.in_channels):
                        for kr in range(k):
                            for kc in range(k):
                                s += x[ic][r+kr][c+kc] * self.W[oc][ic][kr][kc]
                    s += self.b[oc]
                    row_vals.append(s)
                oc_map.append(row_vals)
            out.append(oc_map)
        return out

    def backward(self, grad_out, lr):
        # grad_out shape: out_channels x out_h x out_w
        x = self.last_input
        H = len(x[0])
        W = len(x[0][0])
        k = self.kernel_size
        out_h = len(grad_out[0])
        out_w = len(grad_out[0][0])

        # reset grads
        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                for kr in range(k):
                    for kc in range(k):
                        self.dW[oc][ic][kr][kc] = 0.0
            self.db[oc] = 0.0

        # compute dW, db, and dx
        dx = zeros((self.in_channels, H, W))
        for oc in range(self.out_channels):
            for r in range(out_h):
                for c in range(out_w):
                    go = grad_out[oc][r][c]
                    self.db[oc] += go
                    for ic in range(self.in_channels):
                        for kr in range(k):
                            for kc in range(k):
                                self.dW[oc][ic][kr][kc] += x[ic][r+kr][c+kc] * go
                                dx[ic][r+kr][c+kc] += self.W[oc][ic][kr][kc] * go

        # update weights (SGD)
        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                for kr in range(k):
                    for kc in range(k):
                        self.W[oc][ic][kr][kc] -= lr * self.dW[oc][ic][kr][kc]
            self.b[oc] -= lr * self.db[oc]

        return dx

class TanhLayer:
    def __init__(self):
        self.last_out = None

    def forward(self, x):
        # apply tanh elementwise for 3D or 1D
        if isinstance(x[0], list):
            # 3D (or 2D): I think here it's channels x H x W or 2D matrix
            out = []
            for d in x:
                sub = []
                for row in d:
                    rr = []
                    for v in row:
                        rr.append(tanh(v))
                    sub.append(rr)
                out.append(sub)
            self.last_out = out
            return out
        else:
            out = []
            for v in x:
                out.append(tanh(v))
            self.last_out = out
            return out

    def backward(self, grad_out, lr):
        # I will just multiply by dtanh
        y = self.last_out
        if isinstance(y[0], list):
            out = []
            for ci in range(len(y)):
                sub = []
                for r in range(len(y[ci])):
                    rr = []
                    for c in range(len(y[ci][r])):
                        rr.append(grad_out[ci][r][c] * dtanh(y[ci][r][c]))
                    sub.append(rr)
                out.append(sub)
            return out
        else:
            out = []
            for i in range(len(y)):
                out.append(grad_out[i] * dtanh(y[i]))
            return out

class AvgPool2x2:
    def __init__(self):
        self.last_input = None

    def forward(self, x):
        # x shape: C x H x W, output is C x H/2 x W/2
        self.last_input = x
        C = len(x)
        H = len(x[0])
        W = len(x[0][0])
        out = []
        for c in range(C):
            oc = []
            r = 0
            while r+1 < H:
                row_vals = []
                cidx = 0
                while cidx+1 < W:
                    s = x[c][r][cidx] + x[c][r][cidx+1] + x[c][r+1][cidx] + x[c][r+1][cidx+1]
                    row_vals.append(s / 4.0)
                    cidx += 2
                oc.append(row_vals)
                r += 2
            out.append(oc)
        return out

    def backward(self, grad_out, lr):
        # upsample by distributing the gradient equally to each of the 4 inputs
        x = self.last_input
        C = len(x)
        H = len(x[0])
        W = len(x[0][0])
        dx = zeros((C, H, W))
        for c in range(C):
            r = 0
            go_r = 0
            while r+1 < H:
                cidx = 0
                go_c = 0
                while cidx+1 < W:
                    g = grad_out[c][go_r][go_c]
                    dx[c][r][cidx]     += g * 0.25
                    dx[c][r][cidx+1]   += g * 0.25
                    dx[c][r+1][cidx]   += g * 0.25
                    dx[c][r+1][cidx+1] += g * 0.25
                    cidx += 2
                    go_c += 1
                r += 2
                go_r += 1
        return dx

class Flatten:
    def __init__(self):
        self.shape = None

    def forward(self, x):
        # x is C x H x W
        C = len(x)
        H = len(x[0])
        W = len(x[0][0])
        self.shape = (C, H, W)
        out = []
        for c in range(C):
            for r in range(H):
                for w in range(W):
                    out.append(x[c][r][w])
        return out

    def backward(self, grad_out, lr):
        C, H, W = self.shape
        out = []
        idx = 0
        for c in range(C):
            plane = []
            for r in range(H):
                row = []
                for w in range(W):
                    row.append(grad_out[idx])
                    idx += 1
                plane.append(row)
            out.append(plane)
        return out

class Dense:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        init = he_uniform(in_dim, out_dim)
        self.W = []
        for i in range(out_dim):
            row = []
            for j in range(in_dim):
                row.append(init())
            self.W.append(row)
        self.b = [0.0 for _ in range(out_dim)]
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        out = []
        for i in range(self.out_dim):
            s = 0.0
            for j in range(self.in_dim):
                s += self.W[i][j] * x[j]
            s += self.b[i]
            out.append(s)
        return out

    def backward(self, grad_out, lr):
        x = self.last_input
        dW = []
        for i in range(self.out_dim):
            row = []
            for j in range(self.in_dim):
                row.append(0.0)
            dW.append(row)
        db = [0.0 for _ in range(self.out_dim)]
        dx = [0.0 for _ in range(self.in_dim)]

        for i in range(self.out_dim):
            g = grad_out[i]
            db[i] += g
            for j in range(self.in_dim):
                dW[i][j] += g * x[j]
                dx[j] += self.W[i][j] * g

        # update
        for i in range(self.out_dim):
            self.b[i] -= lr * db[i]
            for j in range(self.in_dim):
                self.W[i][j] -= lr * dW[i][j]

        return dx

class LeNet5Binary:
    # I am going to try to stick close-ish to the original shapes:
    # input 1x32x32
    # C1: conv 6@28x28 (5x5)
    # S2: avgpool -> 6@14x14
    # C3: conv 16@10x10 (5x5)
    # S4: avgpool -> 16@5x5
    # C5: conv 120@1x1 (5x5)
    # F6: dense 84
    # OUTPUT: dense 2 -> softmax
    def __init__(self):
        self.c1 = Conv2D(1, 6, 5)
        self.a1 = TanhLayer()
        self.s2 = AvgPool2x2()
        self.c3 = Conv2D(6, 16, 5)
        self.a3 = TanhLayer()
        self.s4 = AvgPool2x2()
        self.c5 = Conv2D(16, 120, 5)  # will become 120x1x1
        self.a5 = TanhLayer()
        self.flt = Flatten()
        self.f6 = Dense(120, 84)
        self.a6 = TanhLayer()
        self.out = Dense(84, 2)  # 2 classes: coat (0) vs sneaker (1)

    def forward(self, img32):
        # I want to be consistent: img32 is 32x32 list of floats. I will add a channel dimension [1][H][W].
        x = [img32]
        x = self.c1.forward(x)
        x = self.a1.forward(x)
        x = self.s2.forward(x)
        x = self.c3.forward(x)
        x = self.a3.forward(x)
        x = self.s4.forward(x)
        x = self.c5.forward(x)
        x = self.a5.forward(x)  # now 120@1x1
        x = self.flt.forward(x) # 120
        x = self.f6.forward(x)  # 84
        x = self.a6.forward(x)
        logits = self.out.forward(x) # 2
        probs = softmax(logits)
        return logits, probs

    def backward(self, logits, probs, target_index, lr):
        # I am going to do cross-entropy loss with softmax.
        # dL/dlogits = probs; probs[target] -= 1
        grad_logits = []
        for i in range(len(probs)):
            grad_logits.append(probs[i])
        grad_logits[target_index] -= 1.0

        # and then backprop through the layers in reverse order.
        g = self.out.backward(grad_logits, lr)
        g = self.a6.backward(g, lr)
        g = self.f6.backward(g, lr)
        g = self.flt.backward(g, lr)
        g = self.a5.backward(g, lr)
        g = self.c5.backward(g, lr)
        g = self.s4.backward(g, lr)
        g = self.a3.backward(g, lr)
        g = self.c3.backward(g, lr)
        g = self.s2.backward(g, lr)
        g = self.a1.backward(g, lr)
        g = self.c1.backward(g, lr)
        return None

    def save(self, file_path):
        # I am going to save weights in a json file so I can load later.
        data = {
            "c1_W": self.c1.W, "c1_b": self.c1.b,
            "c3_W": self.c3.W, "c3_b": self.c3.b,
            "c5_W": self.c5.W, "c5_b": self.c5.b,
            "f6_W": self.f6.W, "f6_b": self.f6.b,
            "out_W": self.out.W, "out_b": self.out.b
        }
        with open(file_path, 'w') as f:
            import json
            f.write(json.dumps(data))

    def load(self, file_path):
        if not os.path.exists(file_path):
            return False
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
        self.c1.W = data["c1_W"]; self.c1.b = data["c1_b"]
        self.c3.W = data["c3_W"]; self.c3.b = data["c3_b"]
        self.c5.W = data["c5_W"]; self.c5.b = data["c5_b"]
        self.f6.W = data["f6_W"]; self.f6.b = data["f6_b"]
        self.out.W = data["out_W"]; self.out.b = data["out_b"]
        return True
