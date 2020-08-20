Item2vec in Tensorflow
=============
Tensorflow implementation of [Item2Vec: Neural Item Embedding for Collaborative Filtering](https://arxiv.org/vc/arxiv/papers/1603/1603.04259v2.pdf, "paper link"). The referenced code can be found 
[here](https://github.com/carpedm20/word2vec-tensorflow, "ref1 link")
and 
[here](https://github.com/yoonkim/word2vec_torch, "ref2 link").

Prerequisites
-------------

* Python 3.6+
* Tensorflow
* Pickle
* Pandas
* Numpy
* (Optional) Sklearn (for visualization)
* (Optional) Matplotlib (for visualization)

Usage
-------------
Just, run main.py:
<pre>
<code>
$ python main.py
</code>
</pre>

To get most similar item:
<pre>
<code>
Set item numbers. Range is 0 to 300.
10
Nearest to au: av, ax, aw, at, ay, ar, az, as, ap, aq
</code>
</pre>

Visualized Item-vector Using t-SNE
-------------

<img src="https://github.com/werellel/item2vec-Neural-Item-Embedding-For-Collaborative-Filtering/blob/master/t-SNE.png" width="90%"></img>

Author
-------------
[werellel](https://github.com/werellel, "author link")


