import numpy as np
import torch

t1 = torch.tensor([1,2])

print(t1)

print(torch.tensor((1,2)))

a = np.array((1,2))
t2 = torch.tensor(a)
print(t2)

print(a.dtype,t1.dtype,t2.dtype)

t3 = torch.tensor([True,False])
print(t3.dtype)

t4 = torch.tensor([1 +2j,2+ 3j])
print(t4.dtype)

t5 = torch.tensor([1.1,2])
print(t5.dtype)

t6 = torch.tensor([True,2.,False])
print(t6.dtype)
#张量类型的显式转换，不会改变原有张量的类型
print(t1,t1.float(),t1.double(),t1.short())

print(t1.ndim,t1.shape,t1.size())

print(len(t1),t1.numel())

t7 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(t7)

print(t7.ndim,t7.shape,t7.size(),t7.numel(),len(t7))

t8 = torch.tensor(1)
print(t8,t8.ndim,t8.shape,t8.numel())

t9 = torch.tensor([1])
print(t9,t9.ndim,t9.shape,t9.size(),t9.numel())


a1 = np.array([[1,2,3],[4,5,6]])
a2 = np.array([[7,8,9],[4,5,6]])
t10 = torch.tensor(np.array([a1,a2]))
#a1a2两个张量堆叠之后增加了一维
print(a1,a2,t10)

print(t10.ndim,t10.size(),t10.shape,t10.numel(),len(t10))
#flatten 方法可将任意维度张量转化为一维张量
print(t7,t7.flatten())

#reshape 方法任意变形
print(t1,t1.reshape(2,1),t1.shape,t1.reshape(2,1).shape,t1.ndim,t1.reshape(2,1).ndim)

print(t10,t10.flatten())
#转换成一维向量
print(t1,t1.ndim, t1.reshape(2), t1.reshape(2, ), t1.reshape(2).ndim,t1.reshape(2, ).ndim)
#转换成二维向量
print(t1,t1.reshape(1,2),t1.reshape(1,2).ndim)
#转换成三维向量
print(t1,t1.reshape(1,1,2),t1.reshape(1,1,2).shape,t1.reshape(1,1,2).ndim)
#用reshape方法拉平高维度张量
print(t10.reshape(-1))

#3.特殊张量的创建
#全0张量
print(torch.zeros(2,3))
#全1张量
print(torch.ones(2,3))



#Identity matrix
print(torch.eye(6))


#Diagonal matrix
print(t1,torch.diag(t1))

#rand:Obey a uniformly distributed tensor from 0-1
print(torch.rand(2,3))

#randn:Obey the standard normal distribution
print(torch.randn(2,3))

#normal:Obey specified normal distribution(specified mean and std)
print(torch.normal(4,3,size=(2,3)))

#randint:Integer random sampling result(left open right closed)
print(torch.randint(1,100,size = [3,4]))


#arange/linspace:generate a series
print(torch.arange(5),torch.arange(1,10,3))#left open right closed
print(torch.linspace(1,100,5))             #Interval closure

#empty:generate an uninitialized matrix of specified shape
print(torch.empty(2,3))

#full:Fills the specified numeric value based on the specified shape
print(torch.full([5,6],30125))
#specified shape tensor
print(torch.full_like(t1,5))
 

#print(torch.randn_like(t1)) #RuntimeError,_like 
#Type conversion needs to pay attention to the consistency of data types before and after conversion
print(t5,torch.randn_like(t5),t10,torch.randint_like(t10,1,100))

print(t5,torch.zeros_like(t5),torch.ones_like(t5))



#Conversion methods between tensors and other related types

#.numpy methods and np.arrary function:convert tensor to array
print(t1,t1.numpy(),np.array(t1))

#.tolist methods and list function :convert tensor to list
print(t1.tolist(),list(t1))

#.item methods :convert zero-dimensional tensor to numeric values
print(t8,t8.item())



#Deep copy of tensors

#Shallow copy:t12 and t11 both point to the same object
t11 = torch.arange(1,20,2)
t12 = t11
print(t11[1],t11,t12)
t11[1] = 30 
print(t11,t12)


#Deep copy:t13 and t11 point to different objects,equal to create a new object,only the value are same
t13 = t11.clone()
print(t13[3],t13)
t11[3] = 70
print(t11,t12,t13)