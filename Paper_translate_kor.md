[논문 링크](https://arxiv.org/ftp/arxiv/papers/1603/1603.04259.pdf)

https://arxiv.org/ftp/arxiv/papers/1603/1603.04259.pdf

## ITEM2VEC: NEURAL ITEM EMBEDDING FOR COLLABORATIVE FILTERING

많은 Collaborative Filtering 알고리즘은 item과 item 간의 유사성을 생성하기 위해 item-based 방법을 사용한다. (Natural Language Processing)NLP 분야에서 neural embedding algorithms을 이용한 단어의 의미를 학습하는 방법들이 제안되었다. Skip-gram with Negative Sampling  
(SGNS) 또는 word2vec이라고 알려진 방법은 다양한 언어학적 관점에서 최신 기술로 여겨진다.

이 페이퍼에서 item-based CF가 neural word embedding의 동일한 framwork안에서 역할을 맡을 수 있음을 보여준다.

SGNS에 영감을 받아 latent space에 item을 embedding하는 item-based CF를 위한 item2vec을 설명한다.

사용자 정보를 사용할 수 없어도 item간의 관계를 추론할 수 있다. 본 연구에서는 item2vec 방법의 효과를 입증하고 SVD와 비교한다.

### 1\. INTRODUCTION AND RELATED WORK

아이템 유사도 계산은 오늘날 추천 시스템의 핵심 구성 요소이다. 많은 추천 알고리즘은 유저와 아이템의 low demensional embedding 동시에 학습하는데 초점을 두지만 항목의 유사성을 계산하는 것으로 끝이다\[1, 2, 3\]. 아이템 유사성은 온라인 retailter들이 많은 다른 추천 태스크에 광범위하게 사용한다. 본 논문에서는 아이템을 low demensinal space에 embedding하여 아이템의 유사성을 학습 태스크를 다룬다.

item-based 유사성은 온라인 스토어들이 단일 아이템에 기초한 추천을 위해 사용된다. 예를 들어 윈도우 10 앱스토어에 가면 각 앱과 게임들은 "People also like"의 타이틀을 가진 다른 앱 리스트를 가지고 있다. 이 리스트는 전체 페이지에 추천 리스트를 보여줄 수 있다. 아마존이나 켓플리스, 구글플레이스, 아이튠즈 스토어에도 단일 아이템과 비슷한 추천 리스트를 제공한다.

단일 아이템 추천은 특정 아이템에 대한 explicit한 유저 interest와 explicit한 유저의 구매 의향의 context에서 나타나기 때문에 traditional한 user-to-item 추천과는 다르다. 따라서 아이템 유사성에 기반한 단일 아이템 추천은 클릭률이 user-to-item 추천보다 더 높으르모 결과적으로 판매 또는 수익의 더 큰 부분을 담당한다.

아이템 유사성에 기초한 단일 아이템 추천은 다른 추천 태스크에서도 사용된다. bundle 추천은 여러 아이템 집합을 그룹화하여 함께 추천한다. 마지막으로 아이템 유사성은 온라인 스토어의 더 나은 탐색과 UX 향상을 위해 사용된다.

사용자에 대한 slack variables(여유 변수)를 정의하여 아이템 간 연결을 암묵적으로 학습하는 user-item CF 방법이 아이템 관계를 직접 학습하는 방법보다 더 나은 아이템 representation을 만들어 낼 가능성은 낮다.

아이템 유사성은 아이템간의 관계에서 직접 representation을 학습하는 것을 겨냥하는 item-based CF 알고리즘의 핵심이기도 하다\[4, 5\]. 아이템 기반의 CF 방법에는 몇가지 요구사항이 있는데 데이터셋이 많아야 하고 유저 수가 아이템 수 보다 훨씬 많아야 한다. 아이템을 모델링하는 방법의 계산 복잡도는 사용자와 아이템을 동시에 모델링하는 방법보다 현저히 낮다. 예를 들어, 온라인 음악 서비스에는 수만 명의 아티스트를 가진 수억 명의 사용자가 있을 수 있다.

특정 시나리오에서는 user-item 관계를 이용할 수 없다. 예를 들어, 요즘 온라인 샵의 상당 부분은 유저의 명시적인 identification process 없이 이루어진다. 대신 사용 가능한 정보는 세션당 제공된다. 세션을 사용자로 취급하는 것은 엄청나게 비싸고 유익하지 못하다.

최근의 linguistic tasks를 위한 neural embedding method들은 NLP 기능을 극적으로 발전시켰다\[6, 7, 8, 12\]. 이러한 방법은 word와 phrases를 word간의 의미 관계를 캡쳐하는 low dimensional vector space로 map하려고 시도한다. 특히 Skip-gram with Negative Sampling(SGNS), word2vec이라고도 알려진 NLP task\[7, 8\]에서 신기록을 세웠고 그것의 응요프로그램은 NLP를 넘어 다른 도메인에서도 확장되었다\[9. 10\].

이 논문에선 item-based CF에 SGNS를 적용하는 것을 제안한다. 다른 도메인에서의 큰 성공에 동기부여되어, 작은 수정을 거친 SGNS가 CF datasets에서 서로 다른 항목들 사이의 관계를 캡쳐할 수 있다고 제시한다. 이를 위해, 우리는 item2vec이라는 SGNS의 수정된 버전을 제안한다.

본 연구에서는 item2vec가 SVD를 이용하여 item-based CF와 경쟁적인 유사도 측정을 유도할 수 있음을 보여주면서, 다른 보다 복잡한 방법과의 비교를 향후 연구에 맡길 수 있음을 보여준다.

### 2\. SKIP-GRAM WITH NEGATIVE SAMPLING

Skip-gram with negative sampling (SGNS)은 Mikolov et에 의해 도입된 neural word embedding method이다\[8\]. 이 방법은 문장(sentence)에서 단어(word)와 주변 단어 사이의 관계를 포착하는 단어 표현을 찾는 것을 목표로 한다. 이 절의 나머지 부분에서는 SGNS 방법에 대한 간략한 개요를 제공한다.

finite vocabulary $W = {w\_i}\_i^W $에서 sequence of words ${w\_i}\_i^K$가 주어질 때, Skip-gram objective은 아래의 term을 최대화 하는 것에 있다.

$ {1 \\over K } \\sum\_{i=1}^K \\sum\_{-c \\geq k \\geq c, j \\neq 0} log p(w\_{i+j} | w+i)$

equation (1)

c는 context window size($w\_i$에 의존적임)이고 $p(w\_j | w\_i)$ 는 softmax 함수이다.

$p(w\_j | w\_i) = {exp(u\_i^Tv\_j) \\over \\sum exp(u\_i^Tv\_k)}$

equation (2)

$u\_i \\in U (\\subset \\mathbb{R}^m )$와 $v\_i \\in V (\\subset \\mathbb{R}^m )$는 word $w\_i \\in W$를 위한 target 및 context representations에 해당하는 latent vectors이다. $I\_W \\triangleq {1,...,|W|}$와 파라미터 m은 데이터셋의 사이즈에 따라 경험적으로 선택된다.

Eq.(2)는 $\\bigtriangledown p(w\_j | w\_i)$의 vocabulary size |W|가 $10^5 - 10^6$ 사이의 선형 함수이기 때문에 계산 복잡성 때문에 실용적이지 못하다.

Negative Sampling은 Eq.(2)로부터 소프트맥스 함수를 대체함으로써 위의 계산 문제를 완화시킨다.

$p(w\_j | w\_i) = \\sigma(u\_i^T v\_j) \\prod\_{k=1}^N \\sigma(-u\_i^T v\_k)$

여기서 $\\sigma(x) = 1/1+exp(-x)$, N은 positive example 마다 만들어지는 negative sampling의 수를 결정하는 매개변수이다. Negative word $w\_i$는 unigram-distribution$^{3/4}$ 이 distribution은 다른 경험적인 unigram distribution 보다 매우 우수한 성능을 낸다\[8\].

희소한 단어와 빈번한 단어 사이의 불균형을 극복하기 위해서 아래의 subsampling procedure가 제안됐다\[8\]. word sequence가 있을 때 확률 $P(discard | w) = 1 - \\sqrt{\\rho \\over f(w)}$에 의해서 각 단어들이 버려진다. $f(w)$는 단어 w의 빈도를 나타내고 $\\rho$는 규정된 임계값이다.  
이 방법은 학습 프로세스를 가속화하고, 희소한 단어 표현을 우수하게 향상시켰다\[8\].

마지막으로, U와 V는 Eq.(1)에 대해 stochastic gradient ascent(확률적 경사 상승)를 적용함으로써 각각 추정된다.

### 3\. ITEM2VEC – SGNS FOR ITEM SIMILARITY

CF data의 context에서, 아이템은 유저가 생성한 sets으로 주어진다. 유저와 아이템의 집합간의 관계에 대한 정보는 항상 사용할 수 있는 것은 아니다. 예를 들어 주문을 한 유저에 대한 정보없이 store은 주문 데이터셋을 받을 수 있다. 즉 여러 아이템 집합이 동일한 사용자에 속할 수 있는 시나리오가 있지만 이 정보는 제공되지 않는다. 4절에서 이러한 시나리오를 다룬 실험 결과를 제시한다.

우리는 아이템 기반 CF에 SGNS을 적용하고자 한다. SGNS를 CF에 적용하는 것은 일련의 단어가 아이템 집합과 동일하다는 것을 전제로 한다. 따라서 이제부터 '단어'와 '아이템'이라는 용어를 상호교환적으로 사용할 것이다.

sqeunces에서 sets으로 이동함으로써, spatial / time 정보가 손실된다. 우리는 이 정보를 버리기로 선택한다. 왜냐하면 본 논문에서는 유저가 어떤 순서 / 시간에 생성하였든 간에 동일한 집합을 공유하는 아이템이 유사하다고 간주되는 static 환경을 가정한다. 이 가정은 다른 시나리오에서는 포함되지 않을 수 있지만, 우리는 이러한 시나리오의 처리를 본 논문의 범위 밖으로 여긴다.

spatial 정보를 무시하기 때문에, 동일한 sets을 공유하는 각 아이템 쌍을 positive example로 다룬다. 이는 set size로부터 window size가 결정된다는 것을 의미한다. 아이템 집합이 주어질 때 Eq.(1)은 다음과 같이 수정된다.

${1 \\over K} \\sum\_{i=1}^K \\sum\_{j \\neq i}^K log p(w\_j | w\_i)$

또 다른 방법은 Eq.(1)를 유지하며 실행하는 동안 아이템 set을 shuffling 한다. 실험 결과 두 옵션 모두 같은 성능을 낸다. 나머지 프로세스는 섹션 2에서 설명한 방법과 같다. 기술된 방법을 'item2vec'라고 한다.

본 논문에서는 'i-th' 아이템의 최종 표현으로 $u\_i$를 사용하였고, 코사인 유사도를 이용하여 한 쌍의 항목 간의 affinity를 계산하였다. 다른 옵션으로 $v\_i$를 사용하는데 $u\_i + v\_i$ 또는 $\[u\_i^T v\_i^T\]^T$를 사용한다. 마지막 두 옵션은 때때로 우수한 표현을 생성한다.

### 4\. EXPERIMENTAL SETUP AND RESULTS

이번 섹션에서는 item2vec 방법에 대한 실증적(경험적) 평가를 제시한다. 아이템에 대한 메타데이터가 있는지에 따라 질적 및 양적 결과를 제공한다. item-based CF 알고리즘으로 item-item SVD를 사용하였다.

#### 4.1 Datasets

우리는 두개의 다른 데이터셋에서 방법을 평가한다. 첫번째 데이터는 마이크로소프트 Xbox Music 서비스의 유저-아티스트 데이터이다. 이 데이터는 9M 이벤트로 구성되어 있다. 각 이벤트는 유저-아티스트 관계로 구성되어 있다. 즉 유저가 특정 아티스트 음악을 재생했다는 것을 의미한다. 데이터셋에는 732K의 유저와 49K의 구분되는 아티스트가 있다.

두번째 데이터셋은 마이크로소프트 스토어에서 제품의 주문이다. 주문은 유저가 만든 정보에 대한 정보가 없는 basket of item이 주어진다. 따라서 이 데이터셋의 정보는 유저와 아이템을 바인딩할 수 없다는 점에서 취약하다.

데이터셋은 379K개의 주문(아이템 하나 이상 포함)과 1706개의 아이템으로 구성된다.

#### 4.2 Systems and parameters

우리는 item2vec을 두 데이터셋에 적용할 것이다. optimization은 확률적 경사 하강법에 의해 실행된다. 우리는 알고리즘을 20 epochs 실행했다. 우리는 negative sampling value를 both datasets 모두 15로 적용했다. dimension 파라미터 m은 100그리고 40으로 음악과 스토어 데이터에 셋팅했다. 추가적으로 $\\rho$ value를 각각 $10^{-5}$ 그리고 $10^{-3}$ 뮤직과 스토어 데이터셋에 subsampling을 적용했다. 데이터 사이즈가 달라서 다른 파라미터를 설정했다.

우리는 SVD based item-item similarity system과 우리의 방법을 비교한다. 이를 위해 우리는 item 수의 사이즈인 square matrix에 SVD를 적용한다.  
$(i, j)$ entry는 데이터셋에서 positive pair로써 나타나는 횟수$(w\_i, w\_j)$를 포함한다. 그리고 각 entry(항목)을 행과 열 합계의 곱의 제곱근에 따라 정규화했다.

마지막으로, latent representation은 행 $US^{1/2}$에 의해 주어지며, 여기서 S는 대각선이 상위 m의 singular value를 포함하는 대각선 행렬이고, U는 열로 singular vector를 포함하는 행렬이다. 아이템 간의 유사도는 representation의 코사인 유사도에 의해 계산된다.이 절에서 우리는 이 방법을 "SVD"라고 명명한다.

#### 4.3 Experiments and results

음악 데이터셋은 장르에 관한 메타데이터를 제공하지 않는다. 그러므로, 각 아티스트들에 대해 웹에서 장르 메타데이터를 검색하여 장르-아티스트 카탈로그를 구성했다. 그리고 학습된 representation과 장르의 관계를 시각화하기 위해 이 카탈로그를 사용했다. 이것은 유용한 representation이 아티스트의 장르에 따라 클러스터링 될 것이라는 가정에서 온다. 이를 위해 'R&B/Soul', 'Kids' 'Classical', 'Country', 'Electronic/Dance', 'Jazz', 'Latin', 'Hip Hop', '레게/댄스홀', 'Rock', 'World', '기독교/복음서' 등 뚜렷한 장르별로 상위 100명의 인기 아티스트를 수록한 서브셋을 생성했다. 우리는 아이템 벡터의 2차원으로 줄이기 위해 코사인 커널을 t-SNE\[11\]에 적용했다. 그리고 각 아티스트 포인트를 장르에 따라 색칠하였다.

![title](https://www.researchgate.net/profile/Oren_Barkan/publication/298205072/figure/fig2/AS:667889837289510@1536248754885/t-SNE-embedding-for-the-item-vectors-produced-by-Item2Vec-a-and-SVD-b-The-items-are.png)

Fig.2: t-SNE embedding for the item vectors produced by item2vec (a) and SVD (b).  
The items are colored according to a web retrieved genre metadata.

Figures 2(a)와 Figures 2(b)는 item2vec과 SVD에 t-SNE를 적용하여 생성된 2D embedding을 보여준다. 보이는 것 처럼, item2vec 이 더 나은 clustering을 생성한다. 또한 그림에서 비교적 균등한 영역들 중 일부를 관찰했는데 2(a)는 색이 다른 아이템과 섞인 걸 볼 수 있다. 이러한 사례들 중 많은 것들이 웹에서 잘못 표기되거나 혼합된 장르를 가진 예술가들에게서 비롯된다는 것을 발견했다.

TABLE 1: INCONSISTENCIES BETWEEN GENRES FROM  
THE WEB CATALOG AND THE ITEM2VEC BASED KNN  
PREDICTIONS

![title](https://www.researchgate.net/profile/Oren_Barkan/publication/298205072/figure/tbl2/AS:667889837301793@1536248754965/NCONSISTENCIES-BETWEEN-GENRES-FROM-THE-CATALOG-TO-THE-ITEM2VEC-BASED-KNN-PREDICTIONS-K.png)

Table 1은 주어진 아티스트와 관련된 장르가 부정확하거나 적어도 위키피디아와 일관성이 없는 몇 가지 예를 보여준다. 따라서, item2vec와 같은 사용 기반 모델이 잘못된 라벨링된 데이터의 검출에 유용할 수 있으며, 심지어 간단한 k nearest neighbor (KNN) 분류기를 사용하여 올바른 라벨에 대한 제안을 제공할 수 있다고 결론지었다.

TABLE 2: A COMPARISON BETWEEN SVD AND  
ITEM2VEC ON GENRE CLASSIFICATION TASK FOR VARIOUS  
SIZES OF TOP POPULAR ARTIST SETS

![title](https://www.researchgate.net/profile/Oren_Barkan/publication/298205072/figure/tbl1/AS:667889837285389@1536248754944/A-COMPARISON-BETWEEN-SVD-AND-ITEM2VEC-ON-GENRE-CLASSIFICATION-TASK-FOR-VARIOUS-SIZES-OF.png)

TABLE 3: A QUALITATIVE COMPARISON BETWEEN ITEM2VEC AND SVD FOR SELECTED ITEMS FROM THE MUSIC DATASET

[##_Image|kage@8UF3W/btqFUR3xhpM/XrKt9PRTNKTYcZsmDWh9w1/img.png|alignCenter|data-filename="table3.png" data-origin-width="1062" data-origin-height="321" data-ke-mobilestyle="widthContent"|Table 3||_##]

TABLE 4: A QUALITATIVE COMPARISON BETWEEN ITEM2VEC AND SVD FOR SELECTED ITEMS FROM THE STORE DATASET

[##_Image|kage@qCJz8/btqFSZ9ij9s/HcxcvECBq6Ns4gwXtyqnQK/img.png|alignCenter|data-filename="table4.png" data-origin-width="1051" data-origin-height="457" data-ke-mobilestyle="widthContent"|Table 4||_##]

유사도 quality를 정량화하기 위해, 아이템과 가장 가까운 이웃 사이의 장르 일관성을 테스트했다. 우리는 상위 q 인기 항목(다양한 q 값의 경우)을 반복하여 그들의 장르가 그들을 둘러싸고 있는 k 가장 가까운 항목의 장르와 일치하는지 확인한다. 이것은 단순한 다수결 투표로 이루어진다. 우리는 서로 다른 이웃 크기(k = 6, 8, 10, 12, 16)에 대해 동일한 실험을 실시했고, 그 결과 큰 변화는 관찰되지 않았다.

Table 2는 k = 8 일때 얻을 결과를 보여준다. item2vec이 q가 증가함에 따라 SVD모델보다 더 좋다는 것을 볼 수 있다. item2vec이 인기도가 낮은 아이템에 대해 더 나은 representation을 보여준다. item2vec이 인기 있는 아이템을 서브샘플링하고 인기도에 따라 negative sampling을 하기 때문이다. 10K 비인기 아이템의 subset에 동일한 장르 일치성 검정을 적용하여 더 검정해보았다. 인기 없는 종목의 경우 해당 아이템을 이용한 사용자가 15명 미만일 경우 인기없다고 한다. item2vec의 정확도는 68%로 SVD의 58.4%에 비해 높았다.

item2vec과 SVD의 Qualitative 비교는 각각 음악 및 store 데이터셋에 대한 table 3-4에서 볼 수 있다. 테이블에는 씨드 아이템과 가장 가까운 4개가 있다. 이 비교의 주요한 이점은 장르보다 고해상도에서 아이템 유사성을 검사할 수 있는 것이다. 또한 Store 데이터 세트에는 태그 / 레이블이 없기 때문에 질적인 평가를 해야만 한다. 두 데이터 집합 모두에서 'item2vec'는 SVD에서 제공하는 목록보다 시드 아이템과 더 잘 관련된 목록을 제공한다는 것을 볼 수 있다. 또한, 스토어 데이터 세트에 정보가 부족함에도 'item2vec'는 아이템 관계를 상당히 잘 추론하고 있음을 알 수 있다.

### 5\. CONCLUSION

본 논문에서는 아이템 기반 CF를 위한 neural embedding algorithm인 item2vec을 제안했다. item2vec은 SGNS를 기반으로 약간의 수정을 거친다. SVD 기반의 아이템 유사도 모델과 비교했을 때 item2vec의 효과를 입증하는 정량적 평가와 질적 평가를 모두 제시했다.  
item2vec이 기본 SVD 모델에서 얻은 것보다 아이템에 대한 더 나은 representation을 생성하는데, 인기없는 아이템을 더 부각시킨다. item2vec의 인기 있는 아이템의 subsampling과 negative sampling을 사용한다는 사실로 설명할 수 있다. 향후\[1, 2, 3\]과 같은 보다 복잡한 CF 모델을 조사하여 item2vec과 비교할 계획이다. 우리는 항목 유사성의 적용을 위해 SG의 베이지안 변형\[12\]을 연구할 것이다.

### 6\. REFERENCES

\[1\] Paquet, U., Koenigstein, N. (2013, May). One-class  
collaborative filtering with random graphs. In Proceedings of  
the 22nd international conference on World Wide Web (pp.  
999-1008).

\[2\] Koren Y, Bell R, Volinsky C. Matrix factorization  
techniques for recommender systems. Computer. 2009 Aug  
1(8):30-7.

\[3\] Salakhutdinov R, Mnih A. Bayesian probabilistic matrix  
factorization using Markov chain Monte Carlo. In  
Proceedings ICML 2008 Jul 5 (pp. 880-887).

\[4\] Sarwar B, Karypis G, Konstan J, Riedl J. Item-based  
collaborative filtering recommendation algorithms. In  
Proceedings WWW 2001 Apr 1 (pp. 285-295). ACM.

\[5\] Linden G, Smith B, York J. Amazon.com  
recommendations: Item-to-item collaborative filtering.  
Internet Computing, IEEE. 2003 Jan;7(1):76-80.

\[6\] Collobert R, Weston J. A unified architecture for natural  
language processing: Deep neural networks with multitask  
learning. In Proceedings of ICML 2008 Jul 5 (pp. 160-167).

\[7\] Mnih A, Hinton GE. A scalable hierarchical distributed  
language model. In Proceedings of NIPS 2009 (pp. 1081-  
1088).

\[8\] Mikolov T, Sutskever I, Chen K, Corrado GS, Dean J.  
Distributed representations of words and phrases and their  
compositionality. In Proceedings of NIPS 2013 (pp. 3111-  
3119).

\[9\] Frome A, Corrado GS, Shlens J, Bengio S, Dean J,  
Mikolov T. Devise: A deep visual-semantic embedding  
model. Proceedings of NIPS 2013 (pp. 2121-2129).

\[10\] Lazaridou A, Pham NT, Baroni M. Combining language  
and vision with a multimodal skip-gram model. arXiv  
preprint arXiv:1501.02598. 2015 Jan 12.

\[11\] Van der Maaten, L., & Hinton, G. Visualizing data using  
t-SNE. Journal of Machine Learning Research, (2008)  
9(2579-2605), 85.

\[12\] Barkan O. Bayesian neural word embedding. arXiv  
preprint arXiv: 1603.06571. 2015.
