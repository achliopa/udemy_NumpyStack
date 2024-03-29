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

* manual data loading from csv into numpy array is cumbersome
```
for line in open('./data_2d.csv'):
  row = line.split(',')
  sample = map(float, row)
  X.append(sample)
```
* the panda way is easier import then use read_csv `X = pd.read_csv('data_2d.csv',header=None)`
* panda dataframes offer many methods like
	* `.info()` info on dataframe
	* `.head()` first rows, passing a num gives a num of first rows 
* dataframes are molike like DB tables. not 2D numpy arrays
* if we know indexes the easies way to access a specific element is
	* convert dataframe to matrix `M = X.to_numpy()` 
	* access `M[0,0]`
* when we use in pandas `X[0]` we get the column with name 0
* when we use in a 2d matrix `X[0]` in numpy we get first row
* pandas column is of type `pandas.core.series.Series`
* to get a row in a dtaframe we use .iloc[index] which is of Series type
* iloc supports list of rows or range
* we can apply criteria in dataframe seleection `X[X[0] < 5]` means select row where column 0 val is <5
* `X[0]<5` returns an Series of booleans is the selection matrix
* we load in aanew csv `df = pd.read_csv('international-airline-passengers.csv',engine="python",skipfooter=3)`
* in this command we set skipfooter=3 to skip last 2 lines of irrelevant data, also we skip Heders=None as it has header line
* skipfooter does not work with default C engine. so it needs python engine to work
* with `df.columns` we see thje column names.
* to assign new column names `df.columns = ["month","passengers"]`
* we can now retrieve the column by name as `df["month"]` or `df.month`
* to add a new column with all 1s `df["ones"]=1`
* to add a new column where each value is depnedent on the other columns values for this row we use apply and lamda functions
```
df["x1x2"] = df.apply(lambda row: row["x1"]*row["x2"],axis=1)
```
* axis=1 means the method is applied accross each row (vertically) and not across each column (horizontally)
* the above is equivalent to
```
def get)interaction(row):
	return row["x1"]*row["x2"]
df["x1x2"] = df.apply(get_interaction,axis=1)
* to join dataframes on a common column `m = pd.merge(t1,t2,on="user_id")` or `t1.merge(t2,on="user_id")`
```

## Section 4: Matplotlib

* after importing matplotlib and defining the 2 arrays x,y we plot aline chart `plt.plot(x,y)`
* we can add labels to plot
```
plt.xlabel("time")
plt.ylabel("some time method")
plt.title("sin plot")
plt.plot(x,y)
```
* `plt.show()` forces plot to show
* scatterplot `plt.scatter(x,y)` helps find correlations in data
* histogram `plt.hist(x)` shows the distribution of values using buckets
* we can control the number of buckets using param bins `plt.hist(R,bins=20)`
* in linear regresion we get a good fit when the error distance is equaly distributed in a bell shaped histogram
* to plot an image we reshape our matrix to a 2D `im = im.reshape(28,28)` and then plot the image `plt.imshow(im)`
* we can play with colormap `plt.imshow(im, cmap="gray")`
* we can invert the color `plt.imshow(255 - im, cmap="gray")`

## Section 5: Scipy

* the most common distribution is the Gaussian. 
* the Gaussian PDF (probability density function) answers the question. given a sample of a random variable whats the probability
* scipy offers a fast ready method to calc PDF
* we import the lib `from scipy.stats import norm`
* and calc the pdf of 0 in a standardnormal distribution (gausian) `norm.pdf(0)`
* we can pass in params such as mean 'loc' and  standard deviation 'scale' of the gaussian `norm.pdf(0,loc=5,scale=10)`
* variace is standard deviation squared
* we can clac pdf values of an array at same time instead of iteratiing
```
r = np.random.randn(10)
norm.pdf(r)
```
* if we want to calculate the joint probability of some data samples we need the log of Gaussian PDF `norm.logpdf(r)`
* CDF or Cumulative distribution Function is the integral of PDF from -inf to x
* the integral is not solvable, it is calculated numericaly
* scipy has it biltin `norm.cdf(r)`
* cdf has also log CDF `norm.logcdf(r)`
* to sample from a Standard Normal Distribution `r = np.random.randn(1000))`
* to sample from a Gaussian distribution with another mean and standard deviation `r = 10*np.random.randn(10000) + 5`
* we create a 2D gaussian distribution and plot it in a scatteplot
```
r = np.random.randn(10000,2)
plt.scatter(r[:,0],r[:,1])
```
* for an elliptical gaussian (2D gaussian with arbitrary std dev and mean) in one axis
```
r[:,1] = 5*r[:,1]+2 
```
* for producing a multivariate normal distributution using scipy
```
cov = np.array([[1,0.8],[0.8,3]])
from scipy.stats import multivariate_normal as mvn
mu = np.array([0,2])
r = mvn.rvs(mean=mu,cov=cov,size=1000)
```
* same using numpy `r = np.random.multivariate_normal(mean=mu,cov=cov,size=1000)`
* with `scipy.lo.loadmat()` we can load matlab files .mat
* to load sound files (.wav) `scipy.io.wavfile.read()` and .write() to write
* for DSP we heavily use convolution. scipy.signal lib has a whole bunch of methods for DSP
* FFT is also widely used in DSP to go from time domain tot he frequency domain
* we demo it on a siusoidal signal we expect FFT to show 3 spikes on the 3 freqs used
```
x = np.linspace(0,100,10000)
y = np.sin(x) + np.sin(3*x) + np.sin(5*x)
Y = np.fft.fft(y)
```
* FFT gives a signal of complex nums so we need to find magnitude before ploting
```
plt.plot(np.abs(Y))
```