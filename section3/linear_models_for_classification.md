&emsp;&emsp;前面的笔记介绍了三种线性模型，[PLA][1]、[Linear Regression][2]与[Logistic Regression][3]。之所以称他们是线性模型，是因为这三种分类模型的方程中，都含有一个相同的部分，该部分是各个特征的一个线性组合，也可以称这个部分叫做线性评分方程:

$$\color{purple}{s}=w^Tx$$

## 回顾三种线性模型

&emsp;&emsp;严谨一点来说，PLA并不是一种“模型”，PLA (Perceptron Learning Algorithm) 是一种“算法”，用来寻找在“线性可分”的情况下，能够把两个类别完全区分开来的一条直线，所以我们简单的把PLA对应的那个模型就叫做Linear Classification。

&emsp;&emsp;下面对比下这三种模型:

![][4]

- Linear Classification模型：取$\color{purple}{s}$的符号作为结果输出，使用0/1 error作为误差衡量方式，但它的cost function，也就是$E\_{in}(w)$是一个离散的方程，并且该方程的最优化是一个NP-hard问题（简单说就是非常难解的问题）。
- Linear Regression模型：直接输出评分方程，使用平方误差square error作为误差衡量方式，好处是其$E\_{in}(w)$是一个凸二次曲线，非常方便求最优解(可通过矩阵运算一次得到结果)。
- Logistic Regression模型：输出的是评分方程经过sigmoid的结果，使用cross-entropy作为误差衡量方式，其$E\_{in}(w)$是一个凸函数，可以使用gradient descent的方式求最佳解。

&emsp;&emsp;Linear Regression和Logistic Regression的输出是一个实数，而不是一个Binary的值，他们能用来解分类问题吗？可以，只要定一个阈值，高于阈值的输出+1，低于阈值的输出-1就好。既然Linear Regression和Logistic Regression都可以用来解分类问题，并且在最优化上，他们都比Linear Classification简单许多，我们能否使用这两个模型取代Linear Classification呢？

&emsp;&emsp;三个模型的区别在于误差的衡量，误差的衡量可以说是一个模型最重要的部分，这部分内容可以参考[Noise and Error][5]。

![][6]

&emsp;&emsp;这里$y$是一个binary的值，要么是-1，要么是+1。注意到三个模型的error function都有一个$\color{blue}{y}\color{purple}{s}$的部分，也叫做分类正确性分数 (classification correctness score)。其中$\color{purple}{s}$是模型对某个样本给出的分数，$\color{blue}{y}$是该样本的真实值。

&emsp;&emsp;不难看出，当$\color{blue}{y}=+1$时，我们希望$\color{purple}{s}$越大越好，当$\color{blue}{y}=-1$时，我们希望$\color{purple}{s}$越小越好，所以总的来说，我们希望$\color{blue}{y}\color{purple}{s}$尽可能大。因此这里希望给较小的$\color{blue}{y}\color{purple}{s}$较大的cost，给较大的$\color{blue}{y}\color{purple}{s}$较小的cost即可。因此，不同模型的本质差异，就在于这个cost该怎么给。

&emsp;&emsp;既然这三个error function都与$\color{blue}{y}\color{purple}{s}$有关，我们可以以$\color{blue}{y}\color{purple}{s}$为横坐标，$err$为纵坐标，把这三个函数画出来。

![][7]

&emsp;&emsp;sqr (squre error)为Linear Regression的误差函数，ce (cross entropy)为Logistic Regression的误差函数。可以看出，$\color{red}{err\_{sqr}}$在$\color{blue}{y}\color{purple}{s}$较小的时候很大，但是，在$\color{blue}{y}\color{purple}{s}$较大的时候$\color{red}{err\_{sqr}}$同样很大，这点不是很理想，因为我们希望$\color{blue}{y}\color{purple}{s}$大的时候cost要小，尽管如此，至少在$\color{red}{err\_{sqr}}$小的时候，$\color{blue}{err\_{0/1}}$也很小，因此可以拿来做error function。$err\_{ce}$则是一个单调递减的函数，形态有点点像$\color{blue}{err\_{0/1}}$，但来的比较平缓。注意到$err\_{ce}$有一部分是小于$\color{blue}{err\_{0/1}}$的，我们希望$err\_{ce}$能成为$\color{blue}{err\_{0/1}}$的一个upper bound（目的一会儿会说到），只要将$err\_{ce}$做一个换底的动作，即：

$$\color{orange}{\text{scaled}}\text{ ce : err}_{\color{orange}{s}ce}(\color{purple}{s},\color{purple}{y})=\color{orange}{log_2}(1+exp(-\color{purple}{ys}))$$

![][8]

&emsp;&emsp;事实上这里做scale的动作并不会影响最优化的过程，它只是让之后的推导证明更加容易一些。

&emsp;&emsp;现在稍稍回忆一下我们的问题是什么:

&emsp;&emsp;能不能拿Linear Regression或Logistic Regression来替代Linear Classification？

&emsp;&emsp;为什么会想做这样的替代？Linear Classification，在分类这件事上，它做的很好，但在最优化这件事上，由于是NP-hard问题，不大好做，而Linear Regression与Logistic Regression在最优化上比较容易。因此，如果他们在分类能力上的表现能够接近Linear Classification，用他们来替代Linear Classification来处理分类的问题，就是件皆大欢喜的事。这时候就可以想想刚刚为何要把$err\_se$ scale 成$err\_{0/1}$的upper bound，目的就是为了让这几个模型的观点在某个方向上是一致的，即：

&emsp;&emsp;$\color{red}{err\_{sqr}}$/$err\_{sce}$低的时候，$\color{blue}{err\_{0/1}}$也低

&emsp;&emsp;通俗一点讲：

&emsp;&emsp;假设某种疾病有两种检测方法A和B。A方法检查结果为阳性时，则患病，为阴性时，则未患病。B方法的效率差一些，对于一部分患病的人，B方法不一定结果为阳性，但只要B的结果为阳性，再用A来检查，A的结果一定也为阳性。这么一来，我们就可以说，如果B方法的结果为阳性的时候，我们就没有必要使用A方法再检查一次了，它的效率是和A相同的。

&emsp;&emsp;再通俗一点讲：

&emsp;&emsp;如果使用$\color{red}{err\_{sqr}}$/$err\_{sce}$来衡量一个模型分类分得好不好的时候，如果他们认为分得好，那么如果使用$\color{blue}{err\_{0/1}}$，它也会认为分得好。

&emsp;&emsp;对比下在处理分类问题时，使用PLA，Linear Regression以及Logistic Regression的优缺点。

&emsp;&emsp;**PLA**:

- 优点：数据是线性可分时，$E\_{in}^{0/1}$保证可以降到最低
- 缺点：数据不是线性可分时，要额外使用pocket技巧，较难做最优化

&emsp;&emsp;**Linear Regression**:

- 优点：在这三个模型中最容易做最优化
- 缺点：在$\color{blue}{y}\color{purple}{s}$很大或很小时，这个bound是很宽松的，意思就是没有办法保证$E\_{in}^{0/1}$能够很小

&emsp;&emsp;**Logistic Regression**:

- 优点：较容易最优化
- 缺点：当$\color{blue}{y}\color{purple}{s}$是很小的负数时，bound很宽松

&emsp;&emsp;所以我们常常可以使用Linear Regresion跑出的$w$作为(PLA/Pocket/Logistic Regression)的$w\_0$，然后再使用$w\_0$来跑其他模型，这样可以加快其他模型的最优化速度。同时，由于拿到的数据常常是线性不可分的，我们常常会去使用Logistic Regression而不是PLA+pocket。

## Stochastic Gradient Descent

&emsp;&emsp;我们知道PLA与Logistic Regression都是通过迭代的方式来实现最优化的，即：

&emsp;&emsp;For t = 0, 1, ...
$$w_{t+1}\leftarrow w_t + \eta v$$
&emsp;&emsp;when stop, return last w as g

&emsp;&emsp;区别在于，PLA每次迭代只需要针对一个点进行错误修正，而Logistic Regression每一次迭代都需要计算每一个点对于梯度的贡献，再把他们平均起来:

![][9]

&emsp;&emsp;这样一来，数据量大的时候，由于需要计算每一个点，Logistic Regerssion就会很慢了。[上一篇][10]有讲到Logistic Regression每次是怎样迭代的：

<script type="math/tex; mode=display">
w_{t+1} \leftarrow w_t + \eta \underbrace{\color{red}{\frac{1}{N}\sum_{n=1}^{N}}\color{purple}{\theta(\color{black}{-y_nw_t^Tx_n})}\color{orange}{(y_nx_n)}}_{-\color{blue}{\triangledown E_{in}(w_t)}}
</script>

&emsp;&emsp;那么我可以不可以每次只看一个点，即不要公式中先求和再取平均数的那个部分呢？随机取一个点n，它对梯度的贡献为:
$$\color{orange}{\triangledown _w err(w,x_n,y_n)}$$

&emsp;&emsp;我们把它称为随机梯度，stochastic gradient。而真实的梯度，可以认为是随机抽出一个点的梯度值的期望(红色部分取平均数的动作):

<script type="math/tex; mode=display">
\triangledown_w E_{in}(w) = \color{red}{\underset{random\,n}{\epsilon}}\triangledown_w \color{orange}{err(w,x_n,y_n)}
</script>

&emsp;&emsp;因此我们可以把随机梯度当成是在真实梯度上增加一个均值为0的noise：
$$\color{orange}{\text{stochastic gradient}} = \color{blue}{\text{true gradient}} + \color{red}{\text{zero-mean 'noise' directions}}$$

&emsp;&emsp;虽然和true gradient存在一定的误差，但是可以认为在足够多的迭代次数之后，也能达到差不多好的结果。我们把这种方法成为随机梯度下降，Stochastic Gradient Descent (SGD):

<script type="math/tex; mode=display">
w_{t+1} \leftarrow w_t + \eta \underbrace{\color{purple}{\theta(\color{black}{-y_nw_t^Tx_n})}\color{orange}{(y_nx_n)}}_{-\color{blue}{\triangledown_{err}(w_t,x_n,y_n)}}
</script>

&emsp;&emsp;和之前说到的Gradient Descent相比，SGD的好处在于时间复杂度大幅减小(每次只随机地看一个点)，在数据量很大的时候可以很快得得到结果，当然缺点就是，如果前面说到的那个$\color{red}{\text{noise}}$很大的话，会稍稍有点不稳定。

## 多类别分类

&emsp;&emsp;我们现在已经有办法使用线性分类器解决二元分类问题，但有的时候，我们需要对多个类别进行分类，即模型的输出不再是0和1两种，而会是多个不同的类别。那么如何套用二元分类的方法来解决多类别分类的问题呢？

&emsp;&emsp;利用二元分类器来解决多类别分类问题主要有两种策略，OVA(One vs. ALL)和OVO(One vs. One)。

&emsp;&emsp;先来看看OVA，假设原问题有四个类别，那么每次我把其中一个类别当成圈圈，其他所有类别当成叉叉，建立二元分类器，循环下去，最终我们会得到4个分类器。

![][11]

&emsp;&emsp;做预测的时候，分别使用这四个分类器进行预测，预测为圈圈的那个模型所代表的类别，即为最终的输出。譬如正方形的那个分类器输出圈圈，菱形、三角形、星型这三个分类器都说是叉叉，则我们认为它是正方形。当然这里可能遇到一个问题，就是所有模型都说不是自己的时候(都输出叉叉)，怎么办？
&emsp;&emsp;很简单，只要让各个分类器都输出是否为自己类别的概率值，即可，然后选择概率值最高的那个分类器所对应的类别，作为最终的输出。

&emsp;&emsp;在类别较多的时候，如果使用OVA方法，则又会遇到数据不平衡(unbalance)的问题，你拿一个类别作为圈圈，其他所有类别作为叉叉，那么圈圈的比例就会非常小，而叉叉的比例非常高。为了解决这个不平衡的问题，我们可以利用另外一个策略，OVO，即每次只拿两个类别的数据出来建建立分类器，如下图。

![][12]

&emsp;&emsp;这个想法类似在打比赛，一笔新数据进来之后，分别使用这六个模型进行预测，得票数最多的那个类别，作为最终的输出。这样做的好处是，有效率，每次只拿两个类别的数据进行训练，每个模型训练数据量要少很多。但是缺点是，由于模型的数量增加了，将消耗更多的存储空间，会减慢预测的速度。

  [1]: http://beader.me/2013/12/21/perceptron-learning-algorithm/
  [2]: http://beader.me/2014/03/09/linear-regression/
  [3]: http://beader.me/2014/05/03/logistic-regression/
  [4]: images/llloverview.png
  [5]: http://beader.me/2014/03/02/noise-and-error/
  [6]: images/lll_error_function.png
  [7]: images/lll_error_function_vis.png
  [8]: images/lll_error_function_scale.png
  [9]: images/pla_logistic_opt.png
  [10]: http://beader.me/2014/05/03/logistic-regression/
  [11]: images/ova.png
  [12]: images/ovo.png
