# NumPy

NumPy is the fundamental package for all scientific computing in Python.

It's a multi-dimensional array library.

#### Why use NumPy over Lists?
The main difference is the speed. Lists are slow. NumPy is fast because:
* it uses Fixed Type <br>
For example, it stores Integers and ```Int32``` but it can be configure to use ```int8```. Lists, stores a whole lot more data about Integers and requires a lot more space.
* it uses **Contiguous Memory**. In Lists, data are scattered around and they are not in each others vicinity in memory. But in NumPy all memory blocks are next to each other - therefore we can perform **SIMD Vector Processing** on them.

NumPy ships with lots of more functionalities and can perform the similar functions of list in a simpler manner.

For instance, in Lists:

```py
a = [1,3,5]
b = [1,2,3]

a * b
# TypeError: can't multiply sequence by non-int of type 'list'
```

but in NumPy:

```py
a = np.array([1,3,5])
b = np.array([1,2,3])

a * b
# np.array([1,6,15])
```

<br>

### Applications of NumPy

* Mathematics - it can replace MATLAB
* Plotting (Matplotlib)
* Backend (Pandas, Connect 4, Digital Photography)
* Machine Learning - directly and indirectly (Tensors)

<br>

### Load NumPy (in Jupyter)

Install:

```bash
pip install numpy
```

```py
import numpy as np
```
<br>

#### Initialize an Array

```py
a = np.array([1,2,3]) # one-dimensional array
```
```py
b = np.array([[1, 2, 3], [6.0, 5.5, 8]])# two-dimensional array
```
```
[[1.  2.  3. ]
 [6.  5.5 8. ]]
```
```py
thd = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
```
```
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
```

#### Get number of dimensions

```py
b.ndim # 2
```

#### Get shape of array

```py
b.shape # (2, 3) two by three columns
```

#### Get type
To see how much memory an array takes:

```py
a.dtype # dtype('int64')
```

The small integers take 64 bits of memory. To reduce the amount of memory they take we cna define the array as:

```py
c = np.array([1,2,3], dtype='int16')
c.dtype # dtype('int16')
```

#### Get size of array

To get total number of items:

```py
a.size # 3
b.size # 6
c.size # 3
```

To get each item byte size:

```py
a.itemsize # 8 [bytes]
b.itemsize # 8 [bytes]
c.itemsize # 2 [bytes]
```

To get total byte size of an array:

```py
a.nbytes # 24 [bytes]
b.nbytes # 48 [bytes]
c.nbytes # 6 [bytes]
```

#### Accessing specific elements

We can use the ```[row, columns]``` notation:

```py
a[2] # 3rd column = 3
b[1, 0] # 2nd row, 1st column = 6.0
```

Negative elements are counted from last position:

```py
a[-1] # 1st columns from end = 3
```

To get whole row/column data:

```py
b[0, :] # array([1., 2., 3.])
b[:, 0] # array([1., 6.])
```

Or define more settings - [startindex:endindex:stepsize]:

```py
b[0, 0:1:1] # array([1.])
```

#### Assign value to an element(s) by index:

```py
b[0, 1] = 15 # only second item of first row
b[:, 1] = 14 # seconds items of all rows
b[:, 1] = [5, 6] # seconds items of all rows respectively
```

### Initialize different types of array

```py
np.zeros(5) # array([0., 0., 0., 0., 0.])
np.zeros((2, 3, 4))
np.ones((4, 2))
np.ones((4, 2), dtype='int16')
```

To provide the number:

```py
np.full((2,2), 99)
```

And to provide the shape from another array:

```py
np.full_like(a, 99)
```

#### Random numbers

```py
np.random.rand(4,2) # does not require a tuple as an input
np.random.randint(4, size=(3,3)) # only integers
```

### Repeat a shape

```py
arr = np.array([[1,2,3]])
r = np.repeat(arr,3,axis=0)
```
```
[[1 2 3]
 [1 2 3]
 [1 2 3]]
```

### Exercise
Create this array:
```
[[1 1 1 1 1]
 [1 0 0 0 1]
 [1 0 9 0 1]
 [1 0 0 0 1]
 [1 1 1 1 1]]
```

Solution:

```py
arr0s = np.zeros((3,3), dtype='int32')
arr0s[1,1] = 9
arr1s = np.ones((5,5), dtype='int32')
arr1s[1:4, 1:4] = arr0s 
# or arr1s[1:-1, 1:-1] = arr0s 
```

### Arrays are Copied by Reference

```py
ac = np.array([1,2,3])
bc = ac
b[0] = 100
print(a) # [100 2 3]
```

To avoid this use ```copy()``` method:

```py
bc = ac.copy()
```

## Mathematics with NumPy

```py
a = np.array([1,2,3,4])
```
We can perform arithmetic operation on this array:

```py
a + 2 
# array([3, 4, 5, 6])
```

```+``` will add the value to each item. The same thing can eb done with other operators  ```-```, ```*```, ```/```, ```**``` (to power), etc.

We can perform these operations on multiple array as well:

```py
b = np.array([1,0,4,0])
a + b
# array([2, 2, 7, 4])
```

### Trigonometry
is also applicable to arrays:

```py
np.sin(a)
# array([ 0.84147098,  0.90929743,  0.14112001, -0.7568025 ])
```

### Linear Algebra 

#### multiply arrays

```py
arr1 = np.ones((2,3))
arr2 = np.full((3,2), 2)
np.matmul(arr1,arr2)
```
```
array([[6., 6.],
       [6., 6.]])
```

#### Find the determinant

```py
c = np.identity(3)
np.linalg.det(c)
# 1.0
```

### Statistics

```py
stats = np.array([[1,2,3], [4,5,6]])
np.min(stats)
# 1
np.min(stats, axis=1)
# array([1, 4])
```

### Reorganizing Arrays 

#### Reshape

```py
a1 = np.array([[1,2,3, 4], [5,6,7,8]])
a2 = a1.reshape((8, 1))
```
```
array([[1],
       [2],
       [3],
       [4],
       [5],
       [6],
       [7],
       [8]])
```

#### Vertical and Horizontal stacking vectors

```py
v1 = np.array([1,2,3, 4])
v2 = np.array([5,6,7,8])
```
```py
np.vstack([v1,v2])
# array([[1, 2, 3, 4],
#       [5, 6, 7, 8]])
```

Can be done many times like: ```np.vstack([v1,v2,v2])```.

```py
np.hstack([v1,v2])
# array([1, 2, 3, 4, 5, 6, 7, 8])
```

## Load data in from a file

Import file:

```py
filedata = np.genfromtxt('data.txt', delimiter=',')
filedata = filedata.astype('int32')
```

#### Boolean masking

```py
filedata > 50
# or ((filedata > 50) & (filedata < 100))
```
```
array([False, False, True, ....])
```

To get only data above the threshold:
```py
filedata[filedata > 50]
```
```
array([196, 75, 126, ....])
```

#### Other operations - Any, All, etc.

```py
np.any(filedata > 50, axis=0)
np.all(filedata > 50, axis=0)
```


<br>

### References

* [Complete Python NumPy Tutorial](https://www.youtube.com/watch?v=GB9ByFAIAH4)