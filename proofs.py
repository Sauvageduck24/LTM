import torch
import numpy as np
import torch.nn.functional as F

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

x=torch.randn(32,128,768)

q=torch.randn(32,12,128,64)
k=torch.randn(32,12,128,1)
v=torch.randn(32,12,128,64)
h=torch.randn(32,12,128,64)

#posibilidad: 
#q=torch.randn(32,12,128,64)
#k=torch.randn(32,12,128,1)
#v=torch.randn(32,12,128,64)
#h=torch.randn(32,12,128,64)

z=(k+h).transpose(-2,-1)
print(z.shape)
print(q.shape)

final=q @ z / np.sqrt(k.size(-1))

final=F.softmax(final,dim=-1)
final=(final@v).transpose(1,2).contiguous()

print(final.shape)

final=merge_last(final,2)

print()
print(final.shape)
print(x.shape)

final=final+x

print('final: ',final.shape)