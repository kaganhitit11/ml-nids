Poisoning Attacks against Support Vector Machines
Battista Biggio battista.biggio@diee.unica.it
Department of Electrical and Electronic Engineering, University of Cagliari, Piazza d’Armi, 09123 Cagliari, Italy
Blaine Nelson blaine.nelson@wsii.uni-tuebingen.de
Pavel Laskov pavel.laskov@uni-tuebingen.de
Wilhelm Schickard Institute for Computer Science, University of T¨ubingen, Sand 1, 72076 T¨ubingen, Germany
Abstract
We investigate a family of poisoning attacks
against Support Vector Machines (SVM).
Such attacks inject specially crafted train-
ing data that increases the SVM’s test error.
Central to the motivation for these attacks
is the fact that most learning algorithms as-
sume that their training data comes from a
natural or well-behaved distribution. How-
ever, this assumption does not generally hold
in security-sensitive settings. As we demon-
strate, an intelligent adversary can, to some
extent, predict the change of the SVM’s deci-
sion function due to malicious input and use
this ability to construct malicious data.
The proposed attack uses a gradient ascent
strategy in which the gradient is computed
based on properties of the SVM’s optimal so-
lution. This method can be kernelized and
enables the attack to be constructed in the
input space even for non-linear kernels. We
experimentally demonstrate that our gradi-
ent ascent procedure reliably identifies good
local maxima of the non-convex validation er-
ror surface, which significantly increases the
classifier’s test error.
1. Introduction
Machine learning techniques are rapidly emerging as
a vital tool in a variety of networking and large-scale
system applications because they can infer hidden pat-
terns in large complicated datasets, adapt to new be-
haviors, and provide statistical soundness to decision-
making processes. Application developers thus can
Appearing in Proceedings of the 29 th International Confer-
ence on Machine Learning, Edinburgh, Scotland, UK, 2012.
Copyright 2012 by the author(s)/owner(s).
employ learning to help solve so-called big-data prob-
lems and these include a number of security-related
problems particularly focusing on identifying malicious
or irregular behavior. In fact, learning approaches
have already been used or proposed as solutions to
a number of such security-sensitive tasks including
spam, worm, intrusion and fraud detection (Meyer
& Whateley, 2004; Biggio et al., 2010; Stolfo et al.,
2003; Forrest et al., 1996; Bolton & Hand, 2002; Cova
et al., 2010; Rieck et al., 2010; Curtsinger et al., 2011;
Laskov & ˇSrndi´c, 2011). Unfortunately, in these do-
mains, data is generally not only non-stationary but
may also have an adversarial component, and the flex-
ibility afforded by learning techniques can be exploited
by an adversary to achieve his goals. For instance, in
spam-detection, adversaries regularly adapt their ap-
proaches based on the popular spam detectors, and
generally a clever adversary will change his behavior
either to evade or mislead learning.
In response to the threat of adversarial data manip-
ulation, several proposed learning methods explicitly
account for certain types of corrupted data (Globerson
& Roweis, 2006; Teo et al., 2008; Br¨uckner & Schef-
fer, 2009; Dekel et al., 2010). Attacks against learning
algorithms can be classified, among other categories
(c.f. Barreno et al., 2010), into causative (manipula-
tion of training data) and exploratory (exploitation of
the classifier). Poisoning refers to a causative attack in
which specially crafted attack points are injected into
the training data. This attack is especially important
from the practical point of view, as an attacker usually
cannot directly access an existing training database
but may provide new training data; e.g., web-based
repositories and honeypots often collect malware ex-
amples for training, which provides an opportunity for
the adversary to poison the training data. Poisoning
attacks have been previously studied only for simple
anomaly detection methods (Barreno et al., 2006; Ru-
binstein et al., 2009; Kloft & Laskov, 2010).
arXiv:1206.6389v3 [cs.LG] 25 Mar 2013
Poisoning Attacks against SVMs
In this paper, we examine a family of poisoning attacks
against Support Vector Machines (SVM). Following
the general security analysis methodology for machine
learning, we assume that the attacker knows the learn-
ing algorithm and can draw data from the underlying
data distribution. Further, we assume that our at-
tacker knows the training data used by the learner;
generally, an unrealistic assumption, but in real-world
settings, an attacker could instead use a surrogate
training set drawn from the same distribution (e.g.,
Nelson et al., 2008) and our approach yields a worst-
case analysis of the attacker’s capabilities. Under these
assumptions, we present a method that an attacker
can use to construct a data point that significantly
decreases the SVM’s classification accuracy.
The proposed method is based on the properties of
the optimal solution of the SVM training problem.
As was first shown in an incremental learning tech-
nique (Cauwenberghs & Poggio, 2001), this solution
depends smoothly on the parameters of the respective
quadratic programming problem and on the geometry
of the data points. Hence, an attacker can manipu-
late the optimal SVM solution by inserting specially
crafted attack points. We demonstrate that finding
such an attack point can be formulated as optimiza-
tion with respect to a performance measure, subject
to the condition that an optimal solution of the SVM
training problem is retained. Although the test er-
ror surface is generally nonconvex, the gradient ascent
procedure used in our method reliably identifies good
local maxima of the test error surface.
The proposed method only depends on the gradients
of the dot products between points in the input space,
and hence can be kernelized. This contrasts previous
work involving construction of special attack points
(e.g., Br¨uckner & Scheffer, 2009; Kloft & Laskov, 2010)
in which attacks could only be constructed in the fea-
ture space for the nonlinear case. The latter is a
strong disadvantage for the attacker, since he must
construct data in the input space and has no practi-
cal means to access the feature space. Hence, the pro-
posed method breaks new ground in optimizing the im-
pact of data-driven attacks against kernel-based learn-
ing algorithms and emphasizes the need to consider
resistance against adversarial training data as an im-
portant factor in the design of learning algorithms.
2. Poisoning attack on SVM
We assume the SVM has been trained on a data set
Dtr = {xi, yi}n
i=1, xi ∈ Rd. Following the standard no-
tation, K denotes the matrix of kernel values between
two sets of points, Q = yy> K denotes the label-
annotated version of K, and α denotes the SVM’s
dual variables corresponding to each training point.
Depending on the value of αi, the training points are
referred to as margin support vectors (0 < αi < C, set
S), error support vectors (αi = C, set E) and reserve
points (αi = 0, set R). In the sequel, the lower-case
letters s, e, r are used to index the corresponding parts
of vectors or matrices; e.g., Qss denotes the margin
support vector submatrix of Q.
2.1. Main derivation
For a poisoning attack, the attacker’s goal is to find
a point (xc, yc), whose addition to Dtr maximally de-
creases the SVM’s classification accuracy. The choice
of the attack point’s label, yc, is arbitrary but fixed.
We refer to the class of this chosen label as attacking
class and the other as the attacked class.
The attacker proceeds by drawing a validation data
set Dval = {xk, yk}m
k=1 and maximizing the hinge loss
incurred on Dval by the SVM trained on Dtr ∪ (xc, yc):
max
xc
L(xc) =
m∑
k=1
(1 − ykfxc (xk))+ =
m∑
k=1
(−gk)+ (1)
In this section, we assume the role of the attacker and
develop a method for optimizing xc with this objective.
First, we explicitly account for all terms in the margin
conditions gk that are affected by xc:
gk = ∑
j
Qkj αj + ykb − 1 (2)
= ∑
j6 =c
Qkj αj (xc) + Qkc(xc)αc(xc) + ykb(xc) − 1 .
It is not difficult to see from the above equations that
L(xc) is a non-convex objective function. Thus, we
exploit a gradient ascent technique to iteratively op-
timize it. We assume that an initial location of the
attack point x(0)
c has been chosen. Our goal is to up-
date the attack point as x(p)
c = x(p−1)
c + tu where p is
the current iteration, u is a norm-1 vector representing
the attack direction, and t is the step size. Clearly, to
maximize our objective, the attack direction u aligns
to the gradient of L with respect to u, which has to be
computed at each iteration.
Although the hinge loss is not everywhere differen-
tiable, this can be overcome by only considering point
indices k with non-zero contributions to L; i.e., those
for which −gk > 0. Contributions of such points to
the gradient of L can be computed by differentiating
Poisoning Attacks against SVMs
Eq. (2) with respect to u using the product rule:
∂gk
∂u = Qks
∂α
∂u + ∂Qkc
∂u αc + yk
∂b
∂u , (3)
where
∂α
∂u =



∂α1
∂u1 · · · ∂α1
∂ud
... . . . ...
∂αs
∂u1 · · · ∂αs
∂ud


 , simil. ∂Qkc
∂u , ∂b
∂u .
The expressions for the gradient can be further re-
fined using the fact that the step taken in direction
u should maintain the optimal SVM solution. This
can expressed as an adiabatic update condition using
the technique introduced in (Cauwenberghs & Poggio,
2001). Observe that for the i-th point in the training
set, the KKT conditions for the optimal solution of the
SVM training problem can be expressed as:
gi = ∑
j∈Dtr
Qij αj + yib − 1



> 0; i ∈ R
= 0; i ∈ S
< 0; i ∈ E
(4)
h = ∑
j∈Dtr
yj αj = 0 . (5)
The equality in condition (4) and (5) implies that an
infinitesimal change in the attack point xc causes a
smooth change in the optimal solution of the SVM,
under the restriction that the composition of the sets
S, E and R remain intact. This equilibrium allows
us to predict the response of the SVM solution to the
variation of xc, as shown below.
By differentiation of the xc-dependent terms in Eqs.
(4)–(5) with respect to each component ul (1 ≤ l ≤ d),
we obtain, for any i ∈ S,
∂g
∂ul
= Qss
∂α
∂ul
+ ∂Qsc
∂ul
αc + ys
∂b
∂ul
= 0
∂h
∂ul
= y>
s
∂α
∂ul
= 0 ,
(6)
which can be rewritten as
[ ∂b
∂ul
∂α
∂ul
]
= −
[ 0 y>
S
ys Qss
]−1 [ 0
∂Qsc
∂ul
]
αc . (7)
The first matrix can be inverted using the Sherman-
Morrison-Woodbury formula (L¨utkepohl, 1996):
[ 0 y>
s
ys Qss
]−1
= 1
ζ
[−1 υ>
υ ζQ−1
ss − υυ>
]
(8)
where υ = Q−1
ss ys and ζ = y>
s Q−1
ss ys. Substituting
(8) into (7) and observing that all components of the
inverted matrix are independent of xc, we obtain:
∂α
∂u = − 1
ζ αc(ζQ−1
ss − υυ>) · ∂Qsc
∂u
∂b
∂u = − 1
ζ αcυ> · ∂Qsc
∂u .
(9)
Substituting (9) into (3) and further into (1), we obtain
the desired gradient used for optimizing our attack:
∂L
∂u =
m∑
k=1
{
Mk
∂Qsc
∂u + ∂Qkc
∂u
}
αc, (10)
where
Mk = − 1
ζ (Qks(ζQ−1
ss − υυT ) + ykυT ).
2.2. Kernelization
From Eq. (10), we see that the gradient of the objec-
tive function at iteration k may depend on the attack
point x(p)
c = x(p−1)
c + tu only through the gradients of
the matrix Q. In particular, this depends on the cho-
sen kernel. We report below the expressions of these
gradients for three common kernels.
• Linear kernel:
∂Kic
∂u = ∂(xi · x(p)
c )
∂u = txi
• Polynomial kernel:
∂Kic
∂u = ∂(xi · x(p)
c + R)d
∂u = d(xi · x(p)
c + R)d−1txi
• RBF kernel:
∂Kic
∂u = ∂e− γ
2 ||xi−xc||2
∂u = K(xi, x(p)
c )γt(xi − x(p)
c )
The dependence on x(p)
c (and, thus, on u) in the gra-
dients of non-linear kernels can be avoided by substi-
tuting x(p)
c with x(p−1)
c , provided that t is sufficiently
small. This approximation enables a straightforward
extension of our method to arbitrary kernels.
2.3. Poisoning Attack Algorithm
The algorithmic details of the method described in
Section 2.1 are presented in Algorithm 1.
In this algorithm, the attack vector x(0)
c is initialized
by cloning an arbitrary point from the attacked class
and flipping its label. In principle, any point suffi-
ciently deep within the attacking class’s margin can
Poisoning Attacks against SVMs
Algorithm 1 Poisoning attack against SVM
Input: Dtr, the training data; Dval, the validation
data; yc, the class label of the attack point; x(0)
c , the
initial attack point; t, the step size.
Output: xc, the final attack point.
1: {αi, b} ← learn an SVM on Dtr.
2: k ← 0.
3: repeat
4: Re-compute the SVM solution on Dtr ∪{x(p)
c , yc}
using incremental SVM (e.g., Cauwenberghs &
Poggio, 2001). This step requires {αi, b}.
5: Compute ∂L
∂u on Dval according to Eq. (10).
6: Set u to a unit vector aligned with ∂L
∂u .
7: k ← k + 1 and x(p)
c ← x(p−1)
c + tu
8: until L
(
x(p)
c
)
− L
(
x(p−1)
c
)
< 
9: return: xc = x(p)
c
be used as a starting point. However, if this point is
too close to the boundary of the attacking class, the
iteratively adjusted attack point may become a reserve
point, which halts further progress.
The computation of the gradient of the validation error
crucially depends on the assumption that the structure
of the sets S, E and R does not change during the up-
date. In general, it is difficult to determine the largest
step t along an arbitrary direction u, which preserves
this structure. The classical line search strategy used
in gradient ascent methods is not suitable for our case,
since the update to the optimal solution for large steps
may be prohibitively expensive. Hence, the step t is
fixed to a small constant value in our algorithm. After
each update of the attack point x(p)
c , the optimal solu-
tion is efficiently recomputed from the solution on Dtr,
using the incremental SVM machinery (e.g., Cauwen-
berghs & Poggio, 2001).
The algorithm terminates when the change in the vali-
dation error is smaller than a predefined threshold. For
kernels including the linear kernel, the surface of the
validation error is unbounded, hence the algorithm is
halted when the attack vector deviates too much from
the training data; i.e., we bound the size of our attack
points.
3. Experiments
The experimental evaluation presented in the follow-
ing sections demonstrates the behavior of our pro-
posed method on an artificial two-dimensional dataset
and evaluates its effectiveness on the classical MNIST
handwritten digit recognition dataset.
3.1. Artificial data
We first consider a two-dimensional data generation
model in which each class follows a Gaussian distri-
bution with mean and covariance matrices given by
μ− = [−1.5, 0], μ+ = [1.5, 0], Σ− = Σ+ = 0.6I.
The points from the negative distribution are assigned
the label −1 (shown as red in the subsequent figures)
and otherwise +1 (shown as blue). The training and
the validation sets, Dtr and Dval (consisting of 25 and
500 points per class, respectively) are randomly drawn
from this distribution.
In the experiment presented below, the red class is the
attacking class. To this end, a random point of the
blue class is selected and its label is flipped to serve
as the starting point for our method. Our gradient
ascent method is then used to refine this attack un-
til its termination condition is satisfied. The attack’s
trajectory is traced as the black line in Fig. 1 for both
the linear kernel (upper two plots) and the RBF ker-
nel (lower two plots). The background in each plot
represents the error surface explicitly computed for all
points within the box x ∈ [−5, 5]2. The leftmost plots
in each pair show the hinge loss computed on a vali-
dation set while the rightmost plots in each pair show
the classification error for the area of interest. For the
linear kernel, the range of attack points is limited to
the box x ∈ [−4, 4]2 shown as a dashed line.
For both kernels, these plots show that our gradient
ascent algorithm finds a reasonably good local maxi-
mum of the non-convex error surface. For the linear
kernel, it terminates at the corner of the bounded re-
gion, since the error surface is unbounded. For the
RBF kernel, it also finds a good local maximum of the
hinge loss which, incidentally, is the maximum classi-
fication error within this area of interest.
3.2. Real data
We now quantitatively validate the effectiveness of
the proposed attack strategy on a well-known MNIST
handwritten digit classification task (LeCun et al.,
1995). Similarly to Globerson & Roweis (2006), we
focus on two-class sub-problems of discriminating be-
tween two distinct digits.1 In particular, we consider
the following two-class problems: 7 vs. 1; 9 vs. 8; 4
vs. 0. The visual nature of the handwritten digit data
provides us with a semantic meaning for an attack.
Each digit in the MNIST data set is properly normal-
ized and represented as a grayscale image of 28 × 28
pixels. In particular, each pixel is ordered in a raster-
1The data set is also publicly available in Matlab format
at http://cs.nyu.edu/~roweis/data.html.
Poisoning Attacks against SVMsmean Σi ξi (hinge loss)
−5 0 5
−5
0
5
0
0.02
0.04
0.06
0.08
0.1
0.12
0.14
0.16classification error
−5 0 5
−5
0
5
0.01
0.02
0.03
0.04
0.05
0.06mean Σi ξi (hinge loss)
−5 0 5
−5
0
5
0.11
0.115
0.12
0.125
0.13
0.135
0.14
0.145classification error
−5 0 5
−5
0
5
0.02
0.025
0.03
0.035
Figure 1. Behavior of the gradient-based attack strategy on the Gaussian data sets, for the linear (top row) and the RBF
kernel (bottom row) with γ = 0.5. The regularization parameter C was set to 1 in both cases. The solid black line
represents the gradual shift of the attack point x(p)
c toward a local maximum. The hinge loss and the classification error
are shown in colors, to appreciate that the hinge loss provides a good approximation of the classification error. The value
of such functions for each point x ∈ [−5, 5]2 is computed by learning an SVM on Dtr ∪ {x, y = −1} and evaluating its
performance on Dval. The SVM solution on the clean data Dtr, and the training data itself, are reported for completeness,
highlighting the support vectors (with black circles), the decision hyperplane and the margin bounds (with black lines).
scan and its value is directly considered as a feature.
The overall number of features is d = 28 × 28 = 784.
We normalized each feature (pixel value) x ∈ [0, 1]d by
dividing its value by 255.
In this experiment only the linear kernel is considered,
and the regularization parameter of the SVM is fixed
to C = 1. We randomly sample a training and a vali-
dation data of 100 and 500 samples, respectively, and
retain the complete testing data given by MNIST for
Dts. Although it varies for each digit, the size of the
testing data is about 2000 samples per class (digit).
The results of the experiment are presented in Fig. 2.
The leftmost plots of each row show the example of
the attacked class taken as starting points in our algo-
rithm. The middle plots show the final attack point.
The rightmost plots displays the increase in the vali-
dation and testing errors as the attack progresses.
The visual appearance of the attack point reveals that
the attack blurs the initial prototype toward the ap-
pearance of examples of the attacking class. Compar-
ing the initial and final attack points, we see this effect:
the bottom segment of the 7 straightens to resemble
a 1, the lower segment of the 9 becomes more round
thus mimicking an 8, and round noise is added to the
outer boundary of the 4 to make it similar to a 0.
The increase in error over the course of attack is es-
pecially striking, as shown in the rightmost plots. In
general, the validation error overestimates the classifi-
cation error due to a smaller sample size. Nonetheless,
in the exemplary runs reported in this experiment, a
single attack data point caused the classification error
to rise from the initial error rates of 2–5% to 15–20%.
Since our initial attack point is obtained by flipping
the label of a point in the attacked class, the errors
in the first iteration of the rightmost plots of Fig. 2
are caused by single random label flips. This confirms
that our attack can achieve significantly higher error
rates than random label flips, and underscores the vul-
nerability of the SVM to poisoning attacks.
The latter point is further illustrated in a multiple
point, multiple run experiment presented in Fig. 3.
For this experiment, the attack was extended by in-
Poisoning Attacks against SVMsBefore attack (7 vs 1)
5 10 15 20 25
5
10
15
20
25
After attack (7 vs 1)
5 10 15 20 25
5
10
15
20
25
0 200 400
0
0.1
0.2
0.3
0.4
number of iterations
classification error
validation error
testing errorBefore attack (9 vs 8)
5 10 15 20 25
5
10
15
20
25
After attack (9 vs 8)
5 10 15 20 25
5
10
15
20
25
0 200 400
0
0.1
0.2
0.3
0.4
number of iterations
classification error
validation error
testing errorBefore attack (4 vs 0)
5 10 15 20 25
5
10
15
20
25
After attack (4 vs 0)
5 10 15 20 25
5
10
15
20
25
0 200 400
0
0.1
0.2
0.3
0.4
number of iterations
classification error
validation error
testing error
Figure 2. Modifications to the initial (mislabeled) attack point performed by the proposed attack strategy, for the three
considered two-class problems from the MNIST data set. The increase in validation and testing errors across different
iterations is also reported.
jecting additional points into the same class and av-
eraging results over multiple runs on randomly cho-
sen training and validation sets of the same size (100
and 500 samples, respectively). One can clearly see a
steady growth of the attack effectiveness with the in-
creasing percentage of the attack points in the training
set. The variance of the error is quite high, which can
be explained by relatively small sizes of the training
and validation data sets.
4. Conclusions and Future Work
The poisoning attack presented in this paper is the
first step toward the security analysis of SVM against
training data attacks. Although our gradient ascent
method is arguably a crude algorithmic procedure,
it attains a surprisingly large impact on the SVM’s
empirical classification accuracy. The presented at-
tack method also reveals the possibility for assessing
the impact of transformations carried out in the input
space on the functions defined in the Reproducing Ker-
nel Hilbert Spaces by means of differential operators.
Compared to previous work on evasion of learning al-
gorithms (e.g., Br¨uckner & Scheffer, 2009; Kloft &
Laskov, 2010), such influence may facilitate the prac-
tical realization of various evasion strategies. These
implications need to be further investigated.
Several potential improvements to the presented
method remain to be explored in future work. The
first would be to address our optimization method’s
restriction to small changes in order to maintain the
SVM’s structural constraints. We solved this by tak-
Poisoning Attacks against SVMs0 2 4 6 8
0
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
% of attack points in training data
classification error (7 vs 1)
validation error
testing error0 2 4 6 8
0
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
% of attack points in training data
classification error (9 vs 8)
validation error
testing error0 2 4 6 8
0
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
% of attack points in training data
classification error (4 vs 0)
validation error
testing error
Figure 3. Results of the multi-point, multi-run experiments
on the MNIST data set. In each plot, we show the clas-
sification errors due to poisoning as a function of the per-
centage of training contamination for both the validation
(red solid line) and testing sets (black dashed line). The
topmost plot is for the 7 vs.1 classifier, the middle is for
the 9 vs. 8 classifier, and the bottommost is for the 4 vs.
0 classifier.
ing many tiny gradient steps. It would be interesting
to investigate a more accurate and efficient computa-
tion of the largest possible step that does not alter the
structure of the optimal solution.
Another direction for research is the simultaneous opti-
mization of multi-point attacks, which we successfully
approached with sequential single-point attacks. The
first question is how to optimally perturb a subset of
the training data; that is, instead of individually opti-
mizing each attack point, one could derive simultane-
ous steps for every attack point to better optimize their
overall effect. The second question is how to choose
the best subset of points to use as a starting point
for the attack. Generally, the latter is a subset selec-
tion problem but heuristics may allow for improved ap-
proximations. Regardless, we demonstrate that even
non-optimal multi-point attack strategies significantly
degrade the SVM’s performance.
An important practical limitation of the proposed
method is the assumption that the attacker controls
the labels of the injected points. Such assumptions
may not hold when the labels are only assigned by
trusted sources such as humans. For instance, a spam
filter uses its users’ labeling of messages as its ground
truth. Thus, although an attacker can send arbitrary
messages, he cannot guarantee that they will have the
labels necessary for his attack. This imposes an ad-
ditional requirement that the attack data must satisfy
certain side constraints to fool the labeling oracle. Fur-
ther work is needed to understand these potential side
constraints and to incorporate them into attacks.
The final extension would be to incorporate the real-
world inverse feature-mapping problem; that is, the
problem of finding real-world attack data that can
achieve the desired result in the learner’s input space.
For data like handwritten digits, there is a direct map-
ping between the real-world image data and the input
features used for learning. In many other problems
(e.g., spam filtering) the mapping is more complex and
may involve various non-smooth operations and nor-
malizations. Solving these inverse mapping problems
for attacks against learning remains open.
Acknowledgments
This work was supported by a grant awarded to B. Big-
gio by Regione Autonoma della Sardegna, and by
the project No. CRP-18293 funded by the same in-
stitution, PO Sardegna FSE 2007-2013, L.R. 7/2007
“Promotion of the scientific research and technolog-
ical innovation in Sardinia”. The authors also wish
to acknowledge the Alexander von Humboldt Founda-
Poisoning Attacks against SVMs
tion and the Heisenberg Fellowship of the Deutsche
Forschungsgemeinschaft (DFG) for providing financial
support to carry out this research. The opinions ex-
pressed in this paper are solely those of the authors and
do not necessarily reflect the opinions of any sponsor.
References
Barreno, Marco, Nelson, Blaine, Sears, Russell, Joseph,
Anthony D., and Tygar, J. D. Can machine learning be
secure? In Proceedings of the ACM Symposium on Infor-
mation, Computer and Communications Security (ASI-
ACCS), pp. 16–25, 2006.
Barreno, Marco, Nelson, Blaine, Joseph, Anthony D., and
Tygar, J. D. The security of machine learning. Machine
Learning, 81(2):121–148, November 2010.
Biggio, Battista, Fumera, Giorgio, and Roli, Fabio. Multi-
ple classifier systems for robust classifier design in adver-
sarial environments. International Journal of Machine
Learning and Cybernetics, 1(1):27–41, 2010.
Bolton, Richard J. and Hand, David J. Statistical fraud
detection: A review. Journal of Statistical Science, 17
(3):235–255, 2002.
Br¨uckner, Michael and Scheffer, Tobias. Nash equilibria of
static prediction games. In Advances in Neural Informa-
tion Processing Systems (NIPS), pp. 171–179. 2009.
Cauwenberghs, Gert and Poggio, Tomaso. Incremental and
decremental support vector machine learning. In Leen,
T.K., Diettrich, T.G., and Tresp, V. (eds.), Advances in
Neural Information Processing Systems 13, pp. 409–415,
2001.
Cova, M., Kruegel, C., and Vigna, G. Detection and
analysis of drive-by-download attacks and malicious
JavaScript code. In International Conference on World
Wide Web (WWW), pp. 281–290, 2010.
Curtsinger, C., Livshits, B., Zorn, B., and Seifert, C. ZOZ-
ZLE: Fast and precise in-browser JavaScript malware
detection. In USENIX Security Symposium, pp. 33–48,
2011.
Dekel, O., Shamir, O., and Xiao, L. Learning to classify
with missing and corrupted features. Machine Learning,
81(2):149–178, 2010.
Forrest, Stephanie, Hofmeyr, Steven A., Somayaji, Anil,
and Longstaff, Thomas A. A sense of self for unix pro-
cesses. In Proceedings of the IEEE Symposium on Secu-
rity and Privacy, pp. 120–128, 1996.
Globerson, A. and Roweis, S. Nightmare at test time:
Robust learning by feature deletion. In International
Conference on Machine Learning (ICML), pp. 353–360,
2006.
Kloft, Marius and Laskov, Pavel. Online anomaly detection
under adversarial impact. In Proceedings of the 13th
International Conference on Artificial Intelligence and
Statistics (AISTATS), 2010.
Laskov, Pavel and ˇSrndi´c, Nedim. Static detection of ma-
licious JavaScript-bearing PDF documents. In Proceed-
ings of the Annual Computer Security Applications Con-
ference (ACSAC), December 2011.
LeCun, Y., Jackel, L., Bottou, L., Brunot, A., Cortes,
C., Denker, J., Drucker, H., Guyon, I., M¨uller, U.,
S¨ackinger, E., Simard, P., and Vapnik, V. Comparison
of learning algorithms for handwritten digit recognition.
In Int’l Conf. on Artificial Neural Networks, pp. 53–60,
1995.
L¨utkepohl, Helmut. Handbook of matrices. John Wiley &
Sons, 1996.
Meyer, Tony A. and Whateley, Brendon. SpamBayes: Ef-
fective open-source, Bayesian based, email classification
system. In Proceedings of the Conference on Email and
Anti-Spam (CEAS), July 2004.
Nelson, Blaine, Barreno, Marco, Chi, Fuching Jack,
Joseph, Anthony D., Rubinstein, Benjamin I. P., Saini,
Udam, Sutton, Charles, Tygar, J. D., and Xia, Kai. Ex-
ploiting machine learning to subvert your spam filter.
In Proceedings of the 1st USENIX Workshop on Large-
Scale Exploits and Emergent Threats (LEET), pp. 1–9,
2008.
Rieck, K., Kr¨uger, T., and Dewald, A. Cujo: Efficient de-
tection and prevention of drive-by-download attacks. In
Proceedings of the Annual Computer Security Applica-
tions Conference (ACSAC), pp. 31–39, 2010.
Rubinstein, Benjamin I. P., Nelson, Blaine, Huang, Ling,
Joseph, Anthony D., hon Lau, Shing, Rao, Satish, Taft,
Nina, and Tygar, J. D. ANTIDOTE: Understanding
and defending against poisoning of anomaly detectors.
In Proceedings of the 9th ACM SIGCOMM Conference
on Internet Measurement (IMC), pp. 1–14, 2009.
Stolfo, Salvatore J., Hershkop, Shlomo, Wang, Ke,
Nimeskern, Olivier, and Hu, Chia-Wei. A behavior-
based approach to securing email systems. In Math-
ematical Methods, Models and Architectures for Com-
puter Networks Security. Springer-Verlag, 2003.
Teo, C.H., Globerson, A., Roweis, S., and Smola, A. Con-
vex learning with invariances. In Advances in Neural In-
formation Proccessing Systems (NIPS), pp. 1489–1496,
2008.