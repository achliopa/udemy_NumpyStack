# Udemy Course: Tensorflow 2.0: Deep Learning and Artificial Intelligence

* [Course Link](Deep Learning Prerequisites: The Numpy Stack in Python)
* [Course Repo](https://github.com/lazyprogrammer/machine_learning_examples)

## Section 1: Introduction and Outline

* install libs using pip `sudo pip install -U numpy pandas matplotlib scipy`
* install using packages `sudo apt-get install python-numpy python-scipy puthon-matplotlib ipython ipython-notebook python-pandas`

## Section 2: Numpy

* python lists and numpy arrays both support for loops
* python lists support join and append, numpy arrays dont
```
L = [1,2,3]
L.append(4)
L = L + [5]
```
* + in python lists does concatenation
* + in numpy arrays(vectors) does Vector addition (or matrix addition)
* the numpy addition is elementwise
* same holds for * in numpy, in python lists concatenates an identical list
* we cannot raise python lists to power (**) numpy vectors we can
* we numpy we can easily work with matrices. we avoid for loops
* for loops in python are slow
* vector dot product: a . b (column vectors)
	* 1st way: a.b = aTb = S[d=1->D]ad*bd (using vector transpose)
	* 2nd way: a.b = |a||b|cos(theta ab) cos(theta ab) = aTb/|a||b|
* transpose is turn a column to row
* manually calculate dot product
```
dot = 0
for e,f in zip(a,b):
  dot +=e*f
```
* a*b in numpy vectors does elementwise multiplication so an easier way is `np.sum(a*b)`
* or `(a*b).sum()` 
* numpy supports dot method `np.dot(a,b)` or as an instance method `a.dot(b)`
* |a| is the absolute value or length of a vector. we can calc it manually `amag = np.sqrt((a*a).sum())`
* we can use numpy liangebra lib to calc |a| or norm `amag = np.linalg.norm(a)`
* we calc cos(theta ab) `cosangle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))`
* the angle in rad is `angle = np.arccos(cosangle)`
* numpy vector math is 60 times faster than for loops
* A matrix is a >1D array (a list of lists) `M = np.array([[1,2],[3,4]])`
* 1st index is row, 2nd is column
* in vanilla python and numpy in a 2D to access row `M[0]`, to access element `M[0][0]`
* in numpy i can access elements `M[0,0]`
* numpy has a builtin matrix datatype. 
* numpy arrays is the standard also for matrices
* to transpose a matrix `M.T`. in a matrix transpose is invert accross the diagonal
* create an array of zeros passing in size `Z = np.zeros((10,10))`
* same for ones `O = np.ones((5,5))`
* create an array of randoms `R = np.random.random((5,5))`
* for a gaussian (standard distr) random `G = np.random.randn(10,10)`
* numpy overs `.mean()` and `.var()` for the matrices
* For Matrix multiplication inner dimensions must match
* definition C(i,j) = S[k=1->K]A(i,k)B(k,j). (i,j)th entry of C is the dot product of row A(i,:) and column (:,j)
* in numpy Matrix multiplication `C = A.dot(B)`
* its natural to want to do element wise multiplication. arrays must be the same size. the operand is *
* to inverse a matrix `Ainv = np.linalg.inv(A)` Ainv.dot(A) gives the identity matrix
* Matrix determinant (orizousa) `np.linalg.det(A)`
* to get the diagonal of a matrix `np.diag(A)`. we get an 1D array of the diagonal
* if we pass an 1D array to diag() we get a 2D matrix with all zeroes except the diagonal
* Matrixes Outer product (e.g covariance of sample vectors) C(i,j) = A(i,j)B(i,j)
* Dot product is the Inner Product of matrices 
* Outer product in numpy `np.outer(a,b)`
* Inner product in numpy `np.inner(a,b)`
* Matrix trace is the sum of the diagonal `np.trace(A)` equal to `np.diag(A).sum()`
* Eigen Values and Eigen Vectors `np.linalg.eig(A)` or `np.linalg.eigh(C)`
* it returns a tuple of eigen values followed by eigen vectors 
* eigh is for symmetric (A = AT or hermitian Matrixes (A = AH)
* AH  = conjugate transpose of A
* covariance is a symmetric matrix
* to calculate the covariance of a matrix we need to transpose it first `cov = np.cov(X.T)`
* A linear system has the form Ax = b 
* A is a matrix, x is a column vector we try to solve for, b is a vector of nums
* the solution is AinvAx = Ainvb assuminn A is a square matrix (invertible)
* in numpy `x = np.linalg.inv(A).dot(b)` or `x = np.linalg.solve(A,b)`

## Section 3: Pandas

* 