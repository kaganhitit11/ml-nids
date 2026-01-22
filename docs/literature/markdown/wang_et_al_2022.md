
Threats to Training: A Survey of Poisoning Attacks and Defenses on ML Systems 134:11
poisoning attacks. This work creatively gave a direction of the black-box poisoning attack, which
is more practical in real applications.
In spite of the fact that label poisoning attack that merely contaminates a small fraction of
labels of the training data may not independently work for complicated learning paradigms, i.e.,
deep learning, it cannot be ignored that label poisoning is still one of the most basic schemes for
poisoning attacks. Apart from clean label data poisoning method, which is discussed in Section
3.2.4, most methods of data poisoning attacks are still using this technique.
3.1.2 Optimization-based Method. Optimization-based method is another basic scheme for poi-
soning attacks. This method can not only help to compute the best set of data points for label poi-
soning but also can be used to find the most effective scheme for data modification or data injection.
Subsequently, we will shed light on this mainstream method and describe a unified framework to
generalize it.
The core of an optimization-based method is the optimization equation, which concludes the
problem to be solved as a maximization or minimization equation. The workflow of this method
is as follows: (1) the adversaries first transform the poisoning attack problem into an optimization
function to find the global optimal value; (2) then they should use optimization algorithms (e.g.,
gradient descent) to search for solutions under corresponding constraints. Obviously, the perfor-
mance of attacks mainly depends on the construct of optimizations and the strategy to solve the
optimization problem.
It is Nelson et al. [62] who first introduced optimized-based methods into the field of poison-
ing attacks. To make the email system denial of service, the adversary manufactured emails that
maximize the expected spam score of the next legitimate email drawn from the distribution. Later,
Biggio et al. [4] constructed an optimization equation to search the data that can maximize the
classification error in the training set. They alternately updated the victim model and computed
the optimal solution through an iteration algorithm with a gradient ascent strategy. To unify the
iterative procedure of model update and poison data production into an overall framework, Mei
and Zhu [59] first officially put forward the concept of bilevel optimization problem and showed
that the bilevel problem can be solved efficiently using gradient methods. Accordingly, the poison-
ing attack entered a period when researches proposed diverse poisoning methods by revising the
bilevel optimization problem. In this period, these attacks most took place in white-box settings.
Remarkably, the bilevel optimization comprehensively concludes the unified framework for poi-
soning attacks. By changing the choice of objective functions, attack targets, and training dataset,
this framework can be applied to almost all the scenarios and applications for poisoning attacks.
The bilevel optimization can be made explicitly by following Equations (1) and (2):
D∗p ∈arg maxDp
F(Dp,w∗)=L1 (Dval,w∗), (1)
s.t. w∗ ∈arg minw
L2
(Dtrain ∪Dp,w). (2)
As we have predefined in Section 2, Dtrain refers to the original training dataset and Dval
denotes the validation dataset. For brevity, we use Dp to describe the set of poisoned samples. The
objective of the outer optimization (Equation (1)) is to manufacture the most effective poisoned
samples Dp , which maximize the empirical loss function L1(Dval,w∗)on the validation dataset
Dval and the poisoned model w∗. The inner optimization (Equation (2)) is aimed to update the
parameter w∗ of the victim model on the poisoned dataset Dtrain ∪Dp. It should be emphasized
that w∗ depends implicitly on the poisoned samples Dp, so F is defined as a function of two
variables Dval and w∗. The procedure of bilevel optimization is as follows, every time the inner
ACM Computing Surveys, Vol. 55, No. 7, Article 134. Publication date: December 2022.
134:12 Z. Wang et al.
optimization achieves the best local minimization, the outer optimization will newly update the
set of poisoned samples Dp, until the function L1(Dval,w∗)converges.
By appropriately selecting the loss functions L1 in Equation (1) and L2 in Equation (2), the
bilevel framework can encompass diverse poisoning methods against learners for various learning
systems. The loss function L1 is determined by the strategy that the adversary choose, thus largely
determining the performance of the attack. The loss function L2 is decided by the victim model
and the tasks to be solved. For instance, in classification tasks, we usually chooses hinge loss for
binary SVMs [4, 59] and cross-entropy loss for multi-class cases [30, 60].
The original framework merely discusses the white-box setting where poisoning attacks are
conducted by an adversary with full knowledge about the victim model. Actually, the framework
can be easily expanded to the black-box setting and the gray-box setting, as long as the adversary
substitutes the unknown training dataset with surrogate datasets ˆDtrain and ˆDval, sequentially
train a surrogate model ˆM = (ˆDtrain, ˆf , ˆw)in place of the original victim model M = (Dtrain,f ,w)
[46, 96]. Similarly, to conduct untargeted poisoning attack, the original Equation (1) intend to
subvert as many predictions of validation data as possible. However, in targeted attack, adversaries
should replace the whole validation set Dval with targeted test samples, to maximize the impact
of poisons on the set of specific targets [23, 84].
Bilevel optimization provide a unified paradigm for data poisoning. On this basis, to adapt to
broad application scenarios and fast-developing victim models, researchers can continuously im-
prove it and put forward more advanced strategies in future works.
3.1.3 P-tampering Method. The strategy called p-tampering attack allows adversaries to at-
tach malicious adversarial noise to the training data under the bounded budget p, which means 
any incoming training example might be adversarially tampered with independent probability p.
P-tampering attack belongs to targeted poisoning attacks and especially focuses on the online
learning model, such as probably approximately correct (PAC) model [82]. In p-tampering at-
tacks, the adversaries have the capacity of data modification and data injection, but can not modify
the label of any samples.
In 2017, Mahloujifar and Mahmoody [54] proposed a poison-crafting algorithm to bias the av-
erage output of any bounded real-valued function through p-tampering, aiming to increase the
loss of the trained system over a particular test example. However, partial averages of bounded
real-valued functions are not exactly computable in polynomial time. To overcome this drawback,
Mahloujifar et al. [52] presented a new strategy that can efficiently achieve the best bias in polyno-
mial time through approximation within arbitrary small additive error. Another recent work [55]
demonstrated that multi-party learning process might also suffer from poisoning attacks. In any
m-party learning protocol where an adversary controls k (k ∈ [m]) parties with probability p, the 
(k,p)-poisoning attack can increase the probability that the model might fail on a particular target
instance known to the adversary. It is also worth mentioning that these attacks cover the case of
model poisoning in federated learning, since they both allow the distribution of each party in the
multi-party case to be influenced by other parties.
Focusing on the reason why this attack can succeed, Reference [53] discovered the connection
between the general phenomenon of concentration of measure in metric measured spaces and poi-
soning attacks. It proved in theory that for any learning problem defined over such a concentrated
space, no classifier with an initial constant error can be robust to adversarial perturbations.
P-tampering attack proves in theory that PAC learning under such clean-label adversarial noise
is impossible, if the adversary could choose a limited p fraction of tampered examples that he
substitutes with adversarially chosen ones. It also emphasizes the necessity to consider resistance
against adversarial training data as an important factor in the design of ML training.
ACM Computing Surveys, Vol. 55, No. 7, Article 134. Publication date: December 2022.
Threats to Training: A Survey of Poisoning Attacks and Defenses on ML Systems 134:13
3.2 Poisoning on Deep Learning
Along with the broad use of deep learning, attackers have moved their attention to deep learning
instead. The work of Szegedy et al. [79] is the first to attack deep neural networks (DNNs),
which demonstrates the vulnerability of DNNs to surprisingly slight perturbations. When added
to an image, these noises could lead a well-trained DNN to misclassify the adversarial image with
high confidence. Obviously, the weakness also exists in the training phase. However, compared
with conventional machine learning model, which is relatively simpler, deep neural networks are
possessed of more complex structures and more computational demands. Therefore, poisoning
methods mentioned above are insufficient for deep neural networks in most cases.
The main technical difficulty in devising a poisoning attack is the computation of the poisoned
samples, also recently referred to as adversarial training examples. Similar to conventional ma-
chine learning, the poisoning attack on deep learning also focuses on solving a bilevel optimization
problem where the outer optimization amounts to maximizing the loss function on an untainted
validation set, while the inner optimization simulates the training phase on the poisoned training
dataset. Since solving this problem is too computationally demanding, previous works on conven-
tional machine learning have implemented the idea of implicit differentiation in the gradient-based
optimization. This gradient-based method, originating from a solution to compute the discrimi-
nation boundary of SVMs, replaces the inner optimization problem with certain Karush-Kuhn-
Tucker (KKT) conditions to derive an implicit equation for the gradient. This approach however
can only be used against a limited class of learning algorithms, excluding deep learning architec-
tures (i.e., deep neural networks). Due to the inherent complexity of the procedure used to compute
the required gradients, the gradients would likely vanish or explode in such a deep graph. For deep
neural networks, the bilevel objective has to be approximated and improved.
Subsequently, we will introduce several improved methods that are especially designed for deep
learning systems.
3.2.1 Advanced Gradient-based Method. Gradient-based methods generally refer to iterative
solutions to find the global optimal value of a differentiable function step by step. Gradient-based
attacks perturb the training data toward the gradient of the adversarial objective function, until
the poisoned data has the greatest effect. Here, we take the gradient-based solution to the bilevel
optimization (Equations (1) and (2)) as an example. As long as the loss function L1 in Equation (1)
is differentiable, the gradient can be computed with the chain rule:
∇Dp F =∇Dp L1 + ∂w
∂Dp

∇w L1, (3)
where the symbol ∇Dp F indicates the partial derivative of F with respect to Dp. Since the poi-
soned parameter of the victim model w depends implicitly on the poisoned samples Dp, the gra- 
dient ∇Dp F should be computed with the chain rule. Apparently, the challenge in Equation (3) is 
to compute ∂w
∂Dp
, i.e., partial derivative of w with respect to Dp.
Early gradient methods are based on the assumption that the learning problem L2 is convex.
According to KKT conditions, the inner optimization can be replaced with the implicit function
∇w L2(Dtrain ∪Dp,w′)=0. On this basis, Equation (3) can be replaced as follows:
∇Dp F =∇Dp L1 − (∇Dp ∇w L2
)(∇2w L2
)−1 ∇w L1. (4)
Then D(i)
p , the poisoned samples of iteration i, can be updated to D(i+1)
p , the poisoned samples of
next iteration, through methods like gradient ascent:
ACM Computing Surveys, Vol. 55, No. 7, Article 134. Publication date: December 2022.
134:14 Z. Wang et al.
D(i+1)
p =D(i)
p +η∇D(i)p
F (D(i)
p
), (5)
where η refers to the learning rate, which can be preset and adjusted by the adversaries.
Early gradient methods [4, 59] demonstrated above require to compute all the gradients in com-
putational graph and save them in the memory, which results in large demands for storage and
computation resources. Accordingly, advanced gradient-based attacks focus on improving com-
putationally intensive strategies to overcome the limitation of the calculation in poison making.
This method can be used in differentiable optimizations for label poisoning or data modification.
Therefore, adversaries require access to the victim model or a surrogate victim model, which, re-
spectively, corresponds to white-box and black-box settings.
To overcome the limitation of the computational complexity in bilevel optimization on deep
neural networks, Reference [60] exploited a recent technique called the back-gradient method,
which makes it possible to compute the gradient of the inner optimization in a more computa-
tionally efficient and stable manner. Impressively, this is the first poisoning attack able to exploit
neural networks. Jagielski et al. [31] extended this method to linear regression models and present
a theoretically grounded optimization framework for both attacks and defenses specifically tuned
for regression models. To efficiently compute the approximation of the second derivative in Equa-
tion (4), Koh et al. [35] used a combination of fast Hessian-vector products and a conjugate gradient
solver, as well as the back-gradient method to efficiently select the best poisoned points, such that
this simplified solution can work much more efficiently than the original solution to the bilevel
problem. On this basis, Huang et al. [30] solved unique challenges for general-purpose data poison-
ing. With a few stochastic gradient descent (SGD) steps, adversaries crafted poisoned images
that manipulate the victim’s training pipeline to achieve arbitrary model behaviors. Geiping et al.
[23] proposed a poisoning objective that combines the best of both bilevel optimization and heuris-
tic problem through a gradient matching problem, thus ruining the integrity of large-scale deep
learning systems.
With more advanced gradient-based methods (e.g., automatic differentiation, back-gradient, etc.)
emerging, the attack algorithms trace back the entire sequence of parameter updates only when
it is needed to know. Accordingly, compared with previously proposed poisoning strategies, this
approach is targeting more complex learning algorithms like neural networks. However, it is still
challenging to conduct high-volume production of poisoned samples for large DNNs, since the
computing scale will still increase geometrically with the increasing development of network com-
plexity and the growing demand of poisons.
3.2.2 Generative Method. With regard to traditional poisoning attacks, the poisoned data gen-
eration rate is the bottleneck of its implementation. Inspired by the concept of generative models,
researchers introduce generative methods [18, 61, 87, 90], which can bypass the costly gradient
calculation of bilevel optimization and therefore speed up the poisoned data generation.
The key of generative attacks is to train the generative model (e.g., generators and auto-
encoders), which can learn the probability distribution of adversarial perturbations and further
manufacture poisoned samples in a large scale. To generate poisoned samples, the generative
model needs limited knowledge or even full knowledge about the victim model, which, respec-
tively, corresponds to the gray-box setting and the white-box setting.
Yang et al. [87] first constructed an Encoder-Decoder-based framework to accelerate the process
of poison manufacture. As shown in Figure 5(a), this framework is composed of two components,
including the generatorд and the imaginary victim classifier f . In iterationt of the attack, the steps
are as follows: (1) the generator produce poisons xpt ; (2) the adversary inserts the poisons into the
training data and updates the classifier from w(t−1)to w(t); (3) the adversary evaluate the poisoned
model wt on the validation dataset Dval and get the necessary information to guide the direction
ACM Computing Surveys, Vol. 55, No. 7, Article 134. Publication date: December 2022.
Threats to Training: A Survey of Poisoning Attacks and Defenses on ML Systems 134:15
Fig. 5. GAN-based poisoning methods.
for the update of the generator; (4) the adversary updates the generator and goes to the next
iteration. This iterative procedure can be formulated as
д =arg maxд
∑
(x,y)∼Dval
L(fw∗(x),y), (6)
s.t. w∗ =arg minw
∑
(x,y)∼Dp
L(fθ (д (x)),y), (7)
where w indicates the original parameter of f and w∗ refers to the poisoned parameter. The objec-
tive function (Equation (6)) is aimed to produce the generator д, which can manufacture the most
harmful poisons and in turn cause maximum accuracy reduction for the classifier f . The constraint
condition (Equation (7)) here is used to simulate the training process of f on the poisoned dataset
and provide information to update the generator.
On this basis, Feng et al. [18] proposed a similar approach and introduced the pseudo-update
steps when updating the generator. With this trick, adversaries overcome unstable results caused
by direct implementation of alternating updates in the optimization procedure.
In Reference [61], the authors replaced EBGAN with pGAN (Figure 5(b)), which consists of
three components, including the generator G, the discriminator D and the target classifier C(i.e.,
the victim model). Here, the discriminator is designed to distinguish poisons and benign samples
and the generator is aimed to generate poisoned samples, which maximize the error of the target
classifier but minimize the discriminator’s ability to distinguish them from benign data. Compared
with the previous method, the generator in pGAN plays a game against both the discriminator and
the classifier, which can achieve a balance between attack strength and attack concealment.
The use of a generative model allows producing poisoning points at scale, enabling poisoning
attacks against complicated learning algorithms. Additionally, generative methods usually induce
a trade-off between effectiveness and concealment of the attack through the complex game be-
tween classifier and discriminator. In this way, generative methods can flexibly achieve the balance
through weight adjustment on the loss of the discriminator and that of the classifier. Therefore,
these methods can be applied to machine learning classifiers at different risk levels.
3.2.3 Clean-label Method. Different from early poisoning attack scenarios, clean-label attacks
do not require the user to have any control over the labeling process, which is a more realistic
assumption. These attacks generate subtle perturbations to craft poisoned data and then insert
them into the training dataset, without any changes to their labels. Because these poison images
appear to be unmodified, human reviewers will label each example as what it appears to be in
human eyes. The contaminated images often affect classifier behavior on a specific target test
ACM Computing Surveys, Vol. 55, No. 7, Article 134. Publication date: December 2022.
Threats to Training: A Survey of Poisoning Attacks and Defenses on ML Systems 134:31
6.1 Explainable Poisoning
The following questions have puzzled researchers from the very beginning to the very present
during the development of poisoning attacks. When focusing on a certain poison, what kind
of poisoned feature will interfere the model most? When it comes to a large group of poisons,
what kind of poison will twist the decision space of the model to the most extent? As we in-
dicate in Section 3, existing mainstream strategies craft poisons by combining label poisoning
with adversarial perturbation via solving a certain optimization problem. Due to the fact that the
heavy computational expense of solutions to the bilevel optimization greatly block up the applica-
tion of poisoning attack on complicated learning paradigms like deep learning, an essential prob-
lem to be solved is to explore the mechanism behind. Since no universally accepted conjecture
on the existence of how poisoned samples effect machine learning models, some more funda-
mental approaches are required, such as topology, statistics, and other learning theories, which
may accelerate and facilitate the existence of some more threatening attacks and more powerful
defenses.
6.2 Transferable Poisoning
Current poisoning methods mainly focus on white-box attacks, which require adequate knowledge
toward the target victim. On the one hand, it would be empirically impossible for adversaries to get
access to such information in reality without the assistance of any internal spy, which is difficult
to implement in time before every trial. On the other hand, poisons generated by complicated
computation are specifically effective for the certain model but behave poorly on others. As long
as the target model is updated or replaced, the fruits of previous work will turn into waste and the
same hard work has to start all over again. Therefore, inspired by the concept of transferability,
it seems that there is a lot of room for poisoning attack to improve from this perspective. Our
ultimate goal is to conduct effective attacks in black-box and variable settings.
6.3 Accelerated Poisoning
The biggest obstacle in the application of untargeted poisoning attacks, which need more poisons
to wreak yet more damage. Though gradient-based solutions to bi-level optimization acquired cer-
tain effect on traditional machine learning, similar but advanced approaches encounter setbacks in
deep learning algorithms. Faced with rapidly increasing computational complexity, current meth-
ods are sufficient to deal with industrial-scale problems.
6.4 Attacks Toward Other Tasks
Though adopted in such broad applications, the potential of poisoning attack has not been fully
realized. As the rapid replacement of machine learning algorithm, poisoning methods always need
to be upgraded in the face of brand-new and unique challenges.
6.5 Comprehensive Defenses
Normal users usually have no idea about which surface of the ML system will be attacked and
which attack method the adversaries may use. Therefore, it is difficult to exactly select a suitable
defense method with the best effect at once. In addition, new type of poisoning, e.g., clean-label
poisoning, and new scenario of poisoning, e.g., federated learning, invalidate previous approaches
and claim customized defenses. Until now, no defense (including the ones using formal verification)
dares to declare that it is effective against all attacks. Therefore, this issue will remain active and
span several future research directions.
ACM Computing Surveys, Vol. 55, No. 7, Article 134. Publication date: December 2022.
Threats to Training: A Survey of Poisoning Attacks and Defenses on ML Systems 134:33
[6] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. 2017. Machine learning with adversaries:
Byzantine tolerant gradient descent. In Proceedings of the 31st International Conference on Neural Information Processing
Systems. 118–128.
[7] Eitan Borgnia, Valeriia Cherepanova, Liam Fowl, Amin Ghiasi, Jonas Geiping, Micah Goldblum, Tom Goldstein, and
Arjun Gupta. 2021. Strong data augmentation sanitizes poisoning and backdoor attacks without an accuracy tradeoff.
In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP’21). IEEE, 3855–
3859.
[8] Di Cao, Shan Chang, Zhijian Lin, Guohua Liu, and Donghong Sun. 2019. Understanding distributed poisoning attack
in federated learning. In Proceedings of the IEEE 25th International Conference on Parallel and Distributed Systems
(ICPADS’19). IEEE, 233–239.
[9] Alvin Chan, Yi Tay, Yew-Soon Ong, and Aston Zhang. 2020. Poison attacks against text datasets with conditional
adversarially regularized autoencoder. In Proceedings of the Conference on Empirical Methods in Natural Language
Processing: Findings. 4175–4189.
[10] Sen Chen, Minhui Xue, Lingling Fan, Shuang Hao, Lihua Xu, Haojin Zhu, and Bo Li. 2018. Automated poisoning
attacks and defenses in malware detection systems: An adversarial machine learning approach. Comput. Secur. 73
(2018), 326–344.
[11] Yudong Chen, Lili Su, and Jiaming Xu. 2017. Distributed statistical machine learning in adversarial settings: Byzantine
gradient descent. Proceedings of the ACM on Measurement and Analysis of Computing Systems 1, 2 (2017), 1–25.
[12] Gabriela F. Cretu, Angelos Stavrou, Michael E. Locasto, Salvatore J. Stolfo, and Angelos D. Keromytis. 2008. Casting out
demons: Sanitizing training data for anomaly sensors. In Proceedings of the IEEE Symposium on Security and Privacy
(SP’08). IEEE, 81–95.
[13] Jia Ding and Zhiwu Xu. 2020. Adversarial attacks on deep learning models of computer vision: A survey. In Proceedings
of the International Conference on Algorithms and Architectures for Parallel Processing. Springer, 396–408.
[14] Georgios Drainakis, Konstantinos V. Katsaros, Panagiotis Pantazopoulos, Vasilis Sourlas, and Angelos Amditis. 2020.
Federated vs. centralized machine learning under privacy-elastic users: A comparative analysis. In Proceedings of the
IEEE 19th International Symposium on Network Computing and Applications (NCA’20). IEEE, 1–8.
[15] Minghong Fang, Xiaoyu Cao, Jinyuan Jia, and Neil Gong. 2020. Local model poisoning attacks to byzantine-robust
federated learning. In Proceedings of the 29th USENIX Security Symposium (USENIX Security’20). 1605–1622.
[16] Minghong Fang, Neil Zhenqiang Gong, and Jia Liu. 2020. Influence function-based data poisoning attacks to top-n
recommender systems. In Proceedings of the Web Conference 2020. 3019–3025.
[17] Minghong Fang, Guolei Yang, Neil Zhenqiang Gong, and Jia Liu. 2018. Poisoning attacks to graph-based recommender
systems. In Proceedings of the 34th Annual Computer Security Applications Conference. 381–392.
[18] Ji Feng, Qi-Zhi Cai, and Zhi-Hua Zhou. 2019. Learning to confuse: Generating training time adversarial data with
auto-encoder. Adv. Neural Info. Process. Syst. 32 (2019), 11994–12004.
[19] Jiashi Feng, Huan Xu, Shie Mannor, and Shuicheng Yan. 2014. Robust logistic regression and classification. Adv. Neural
Info. Process. Syst. 27 (2014), 253–261.
[20] Adriano Franci, Maxime Cordy, Martin Gubri, Mike Papadakis, and Yves Le Traon. 2020. Effective and efficient data
poisoning in semi-supervised learning. Retrieved from https://arXiv:2012.07381.
[21] Benoît Frénay and Michel Verleysen. 2013. Classification in the presence of label noise: A survey. IEEE Trans. Neural
Netw. Learn. Syst. 25, 5 (2013), 845–869.
[22] Shuhao Fu, Chulin Xie, Bo Li, and Qifeng Chen. 2019. Attack-resistant federated learning with residual-based reweight-
ing. Retrieved from https://arXiv:1912.11464.
[23] Jonas Geiping, Liam H. Fowl, W. Ronny Huang, Wojciech Czaja, Gavin Taylor, Michael Moeller, and Tom Goldstein.
2020. Witches’ Brew: Industrial scale data poisoning via gradient matching. In Proceedings of the International Confer-
ence on Learning Representations.
[24] Micah Goldblum, Dimitris Tsipras, Chulin Xie, Xinyun Chen, Avi Schwarzschild, Dawn Song, Aleksander Madry, Bo
Li, and Tom Goldstein. 2022. Dataset security for machine learning: Data poisoning, backdoor attacks, and defenses.
IEEE Trans. Pattern Anal. Mach. Intell. 99, 1 (2022), 1–18.
[25] Rachid Guerraoui, Sébastien Rouault, et al. 2018. The hidden vulnerability of distributed learning in byzantium. In
Proceedings of the International Conference on Machine Learning. PMLR, 3521–3530.
[26] Hao Guo, Brian Dolhansky, Eric Hsin, Phong Dinh, Cristian Canton Ferrer, and Song Wang. 2021. Deep poisoning:
Towards robust image data sharing against visual disclosure. In Proceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision. 686–696.
[27] Junfeng Guo and Cong Liu. 2020. Practical poisoning attacks on neural networks. In Proceedings of the European
Conference on Computer Vision. Springer, 142–158.
[28] Dan Hendrycks, Mantas Mazeika, Duncan Wilson, and Kevin Gimpel. 2018. Using trusted data to train deep networks
on labels corrupted by severe noise. Adv. Neural Info. Process. Syst. 31 (2018), 10456–10465.
ACM Computing Surveys, Vol. 55, No. 7, Article 134. Publication date: December 2022.
134:34 Z. Wang et al.
[29] Rui Hu, Yuanxiong Guo, Miao Pan, and Yanmin Gong. 2019. Targeted poisoning attacks on social recommender sys-
tems. In Proceedings of the IEEE Global Communications Conference (GLOBECOM’19). IEEE, 1–6.
[30] W. Ronny Huang, Jonas Geiping, Liam Fowl, Gavin Taylor, and Tom Goldstein. 2020. MetaPoison: Practical general-
purpose clean-label data poisoning. Adv. Neural Info. Process. Syst. 33 (2020), 12080–12091.
[31] Matthew Jagielski, Alina Oprea, Battista Biggio, Chang Liu, Cristina Nita-Rotaru, and Bo Li. 2018. Manipulating ma-
chine learning: Poisoning attacks and countermeasures for regression learning. In Proceedings of the IEEE Symposium
on Security and Privacy (SP’18). IEEE, 19–35.
[32] Jinyuan Jia, Xiaoyu Cao, and Neil Zhenqiang Gong. 2021. Intrinsic certified robustness of bagging against data poi-
soning attacks. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35. 7961–7969.
[33] Michael I. Jordan and Tom M. Mitchell. 2015. Machine learning: Trends, perspectives, and prospects. Science 349, 6245
(2015), 255–260.
[34] Pang Wei Koh and Percy Liang. 2017. Understanding black-box predictions via influence functions. In Proceedings of
the International Conference on Machine Learning. PMLR, 1885–1894.
[35] Pang Wei Koh, Jacob Steinhardt, and Percy Liang. 2018. Stronger data poisoning attacks break data sanitization de-
fenses. Retrieved from https://arXiv:1811.00741.
[36] Nikola Konstantinov and Christoph Lampert. 2019. Robust learning from untrusted sources. In Proceedings of the
International Conference on Machine Learning. PMLR, 3488–3498.
[37] Moshe Kravchik, Battista Biggio, and Asaf Shabtai. 2021. Poisoning attacks on cyber attack detectors for industrial
control systems. In Proceedings of the 36th Annual ACM Symposium on Applied Computing. 116–125.
[38] Keita Kurita, Paul Michel, and Graham Neubig. 2020. Weight poisoning attacks on pretrained models. In Proceedings
of the 58th Annual Meeting of the Association for Computational Linguistics. 2793–2806.
[39] Alexander Levine and Soheil Feizi. 2020. Deep partition aggregation: Provable defenses against general poisoning
attacks. In Proceedings of the International Conference on Learning Representations.
[40] Bo Li, Yining Wang, Aarti Singh, and Yevgeniy Vorobeychik. 2016. Data poisoning attacks on factorization-based
collaborative filtering. Adv. Neural Info. Process. Syst. 29 (2016), 1885–1893.
[41] Suyi Li, Yong Cheng, Wei Wang, Yang Liu, and Tianjian Chen. 2020. Learning to detect malicious clients for robust
federated learning. Retrieved from https://arXiv:2002.00211.
[42] Yuncheng Li, Jianchao Yang, Yale Song, Liangliang Cao, Jiebo Luo, and Li-Jia Li. 2017. Learning from noisy labels with
distillation. In Proceedings of the IEEE International Conference on Computer Vision. 1910–1918.
[43] Chang Liu, Bo Li, Yevgeniy Vorobeychik, and Alina Oprea. 2017. Robust linear regression against training data poi-
soning. In Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security. 91–102.
[44] Fang Liu and Ness Shroff. 2019. Data poisoning attacks on stochastic bandits. In Proceedings of the International Con-
ference on Machine Learning. PMLR, 4042–4050.
[45] Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. 2018. Fine-pruning: Defending against backdooring attacks on
deep neural networks. In Proceedings of the International Symposium on Research in Attacks, Intrusions, and Defenses.
Springer, 273–294.
[46] Sijia Liu, Songtao Lu, Xiangyi Chen, Yao Feng, Kaidi Xu, Abdullah Al-Dujaili, Mingyi Hong, and Una-May O¡ ̄Reilly.
2020. Min-max optimization without gradients: Convergence and applications to black-box evasion and poisoning
attacks. In Proceedings of the International Conference on Machine Learning. PMLR, 6282–6293.
[47] Xuanqing Liu, Si Si, Xiaojin Zhu, Yang Li, and Cho-Jui Hsieh. 2019. A unified framework for data poisoning attack
to graph-based semi-supervised learning. In Proceedings of the 33rd International Conference on Neural Information
Processing Systems. 9780–9790.
[48] Yi Liu, Xingliang Yuan, Ruihui Zhao, Yifeng Zheng, and Yefeng Zheng. 2020. RC-SSFL: Towards robust and
communication-efficient semi-supervised federated learning system. Retrieved from https://arXiv:2012.04432.
[49] Yuzhe Ma, Kwang-Sung Jun, Lihong Li, and Xiaojin Zhu. 2018. Data poisoning attacks in contextual bandits. In Pro-
ceedings of the International Conference on Decision and Game Theory for Security. Springer, 186–204.
[50] Yuzhe Ma, Xuezhou Zhang, Wen Sun, and Xiaojin Zhu. 2019. Policy poisoning in batch reinforcement learning and
control. In Proceedings of the 33rd International Conference on Neural Information Processing Systems. 14570–14580.
[51] Yuzhe Ma, Xiaojin Zhu Zhu, and Justin Hsu. 2019. Data poisoning against differentially private learners: Attacks and
defenses. In Proceedings of the International Joint Conference on Artificial Intelligence. 4732–4738.
[52] Saeed Mahloujifar, Dimitrios I. Diochnos, and Mohammad Mahmoody. 2018. Learning under p-tampering attacks. In
Algorithmic Learning Theory. PMLR, 572–596.
[53] Saeed Mahloujifar, Dimitrios I. Diochnos, and Mohammad Mahmoody. 2019. The curse of concentration in robust
learning: Evasion and poisoning attacks from concentration of measure. In Proceedings of the AAAI Conference on
Artificial Intelligence, Vol. 33. 4536–4543.
[54] Saeed Mahloujifar and Mohammad Mahmoody. 2017. Blockwise p-tampering attacks on cryptographic primitives,
extractors, and learners. In Proceedings of the Theory of Cryptography Conference. Springer, 245–279.
ACM Computing Surveys, Vol. 55, No. 7, Article 134. Publication date: December 2022.
Threats to Training: A Survey of Poisoning Attacks and Defenses on ML Systems 134:35
[55] Saeed Mahloujifar, Mohammad Mahmoody, and Ameer Mohammed. 2019. Data poisoning attacks in multi-party learn-
ing. In Proceedings of the International Conference on Machine Learning. PMLR, 4274–4283.
[56] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. Communication-
efficient learning of deep networks from decentralized data. In Artificial Intelligence and Statistics. PMLR, 1273–1282.
[57] Akshay Mehra, Bhavya Kailkhura, Pin-Yu Chen, and Jihun Hamm. 2021. How robust are randomized smoothing-based
defenses to data poisoning? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
13244–13253.
[58] Akshay Mehra, Bhavya Kailkhura, Pin-Yu Chen, and Jihun Hamm. 2021. Understanding the limits of unsupervised
domain adaptation via data poisoning. Retrieved from https://arXiv:2107.03919.
[59] Shike Mei and Xiaojin Zhu. 2015. Using machine teaching to identify optimal training-set attacks on machine learners.
In Proceedings of the 29th AAAI Conference on Artificial Intelligence. 2871–2877.
[60] Luis Muñoz-González, Battista Biggio, Ambra Demontis, Andrea Paudice, Vasin Wongrassamee, Emil C. Lupu, and
Fabio Roli. 2017. Towards poisoning of deep learning algorithms with back-gradient optimization. In Proceedings of
the 10th ACM Workshop on Artificial Intelligence and Security. 27–38.
[61] Luis Muñoz-González, Bjarne Pfitzner, Matteo Russo, Javier Carnerero-Cano, and Emil C. Lupu. 2019. Poisoning at-
tacks with generative adversarial nets. Retrieved from https://arXiv:1906.07773.
[62] Blaine Nelson, Marco Barreno, Fuching Jack Chi, Anthony D. Joseph, Benjamin I. P. Rubinstein, Udam Saini, Charles
Sutton, J. Doug Tygar, and Kai Xia. 2008. Exploiting machine learning to subvert your spam filter. In Proceedings of
First USENIX Workshop on Large Scale Exploits and Emergent Threats (LEET’08). 1–9. 
[63] Tribhuvanesh Orekondy, Bernt Schiele, and Mario Fritz. 2020. Prediction poisoning: Towards defenses against DNN
model stealing attacks. In Proceedings of the 8th International Conference on Learning Representations.
[64] Naman Patel, Prashanth Krishnamurthy, Siddharth Garg, and Farshad Khorrami. 2020. Bait and switch: Online train-
ing data poisoning of autonomous driving systems. Retrieved from https://arXiv:2011.04065.
[65] Andrea Paudice, Luis Muñoz-González, and Emil C. Lupu. 2018. Label sanitization against label flipping poisoning
attacks. In Proceedings of the Joint European Conference on Machine Learning and Knowledge Discovery in Databases.
Springer, 5–15.
[66] Neehar Peri, Neal Gupta, W. Ronny Huang, Liam Fowl, Chen Zhu, Soheil Feizi, Tom Goldstein, and John P. Dickerson.
2020. Deep k-NN defense against clean-label data poisoning attacks. In Proceedings of the European Conference on
Computer Vision. Springer, 55–70.
[67] Amin Rakhsha, Goran Radanovic, Rati Devidze, Xiaojin Zhu, and Adish Singla. 2020. Policy teaching via environ-
ment poisoning: Training-time adversarial attacks against reinforcement learning. In Proceedings of the International
Conference on Machine Learning. PMLR, 7974–7984.
[68] Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urtasun. 2018. Learning to reweight examples for robust deep
learning. In Proceedings of the International Conference on Machine Learning. PMLR, 4334–4343.
[69] Mauro Ribeiro, Katarina Grolinger, and Miriam A. M. Capretz. 2015. Mlaas: Machine learning as a service. In Proceed-
ings of the IEEE 14th International Conference on Machine Learning and Applications (ICMLA’15). IEEE, 896–902.
[70] Yuji Roh, Kangwook Lee, Steven Whang, and Changho Suh. 2020. Fr-train: A mutual information-based approach to
fair and robust training. In Proceedings of the International Conference on Machine Learning. PMLR, 8147–8157.
[71] Elan Rosenfeld, Ezra Winston, Pradeep Ravikumar, and Zico Kolter. 2020. Certified robustness to label-flipping attacks
via randomized smoothing. In Proceedings of the International Conference on Machine Learning. PMLR, 8230–8241.
[72] Roei Schuster, Tal Schuster, Yoav Meri, and Vitaly Shmatikov. 2020. Humpty dumpty: Controlling word meanings via
corpus poisoning. In Proceedings of the IEEE Symposium on Security and Privacy (SP’20). IEEE, 1295–1313.
[73] Roei Schuster, Congzheng Song, Eran Tromer, and Vitaly Shmatikov. 2021. You autocomplete me: Poisoning vul-
nerabilities in neural code completion. In Proceedings of the 30th USENIX Security Symposium (USENIX Security’21).
1559–1575.
[74] Alex Serban, Erik Poll, and Joost Visser. 2020. Adversarial examples on object recognition: A comprehensive survey.
ACM Comput. Surveys 53, 3 (2020), 1–38.
[75] Ali Shafahi, W. Ronny Huang, Mahyar Najibi, Octavian Suciu, Christoph Studer, Tudor Dumitras, and Tom Goldstein.
2018. Poison frogs! targeted clean-label poisoning attacks on neural networks. In Proceedings of the 32nd International
Conference on Neural Information Processing Systems. 6106–6116.
[76] Shiqi Shen, Shruti Tople, and Prateek Saxena. 2016. Auror: Defending against poisoning attacks in collaborative deep
learning systems. In Proceedings of the 32nd Annual Conference on Computer Security Applications. 508–519.
[77] Jacob Steinhardt, Pang Wei Koh, and Percy Liang. 2017. Certified defenses for data poisoning attacks. In Proceedings
of the 31st International Conference on Neural Information Processing Systems. 3520–3532.
[78] Yanchao Sun, Da Huo, and Furong Huang. 2020. Vulnerability-aware poisoning mechanism for online RL with un-
known dynamics. In Proceedings of the International Conference on Learning Representations.
ACM Computing Surveys, Vol. 55, No. 7, Article 134. Publication date: December 2022.
134:36 Z. Wang et al.
[79] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus.
2014. Intriguing properties of neural networks. In Proceedings of the 2nd International Conference on Learning Repre-
sentations (ICLR’14).
[80] Vale Tolpegin, Stacey Truex, Mehmet Emre Gursoy, and Ling Liu. 2020. Data poisoning attacks against federated
learning systems. In Proceedings of the European Symposium on Research in Computer Security. Springer, 480–501.
[81] Brandon Tran, Jerry Li, and Aleksander Madry. 2018. Spectral signatures in backdoor attacks. In Proceedings of the
32nd International Conference on Neural Information Processing Systems. 8011–8021.
[82] Leslie G. Valiant. 1984. A theory of the learnable. Commun. ACM 27, 11 (1984), 1134–1142.
[83] Andreas Veit, Neil Alldrin, Gal Chechik, Ivan Krasin, Abhinav Gupta, and Serge Belongie. 2017. Learning from noisy
large-scale datasets with minimal supervision. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition. 839–847.
[84] Huang Xiao, Battista Biggio, Gavin Brown, Giorgio Fumera, Claudia Eckert, and Fabio Roli. 2015. Is feature selection
secure against training data poisoning? In Proceedings of the International Conference on Machine Learning. PMLR, 
1689–1698.
[85] Han Xiao, Huang Xiao, and Claudia Eckert. 2012. Adversarial label flips attack on support vector machines. In Pro-
ceedings of the 20th European Conference on Artificial Intelligence (ECAI’12). IOS Press, 870–875.
[86] Han Xu, Yao Ma, Hao-Chen Liu, Debayan Deb, Hui Liu, Ji-Liang Tang, and Anil K. Jain. 2020. Adversarial attacks and
defenses in images, graphs and text: A review. Int. J. Autom. Comput. 17, 2 (2020), 151–178.
[87] Chaofei Yang, Qing Wu, Hai Li, and Yiran Chen. 2017. Generative poisoning attack method against neural networks.
Retrieved from https://arXiv:1703.01340.
[88] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. 2018. Byzantine-robust distributed learning: To-
wards optimal statistical rates. In Proceedings of the International Conference on Machine Learning. PMLR, 5650–5659.
[89] Xiaoyong Yuan, Pan He, Qile Zhu, and Xiaolin Li. 2019. Adversarial examples: Attacks and defenses for deep learning.
IEEE Trans. Neural Netw. Learn. Syst. 30, 9 (2019), 2805–2824.
[90] Jiale Zhang, Junjun Chen, Di Wu, Bing Chen, and Shui Yu. 2019. Poisoning attack in federated learning using gen-
erative adversarial nets. In Proceedings of the 18th IEEE International Conference On Trust, Security And Privacy In
Computing And Communications/13th IEEE International Conference On Big Data Science And Engineering (Trust-
Com/BigDataSE)’19. IEEE, 374–380.
[91] Rui Zhang and Quanyan Zhu. 2017. A game-theoretic analysis of label flipping attacks on distributed support vector
machines. In Proceedings of the 51st Annual Conference on Information Sciences and Systems (CISS’17). IEEE, 1–6.
[92] Xuezhou Zhang, Xiaojin Zhu, and Laurent Lessard. 2020. Online data poisoning attacks. In Learning for Dynamics and
Control. PMLR, 201–210.
[93] Lingchen Zhao, Shengshan Hu, Qian Wang, Jianlin Jiang, Shen Chao, Xiangyang Luo, and Pengfei Hu. 2020. Shielding
collaborative learning: Mitigating poisoning attacks through client-side detection. IEEE Trans. Depend. Secure Comput.
18, 5 (2020), 2029–2041.
[94] Mengchen Zhao, Bo An, Wei Gao, and Teng Zhang. 2017. Efficient label contamination attacks against black-box
learning models. In Proceedings of the 26th International Joint Conference on Artificial Intelligence. 3945–3951.
[95] Mengchen Zhao, Bo An, Yaodong Yu, Sulin Liu, and Sinno Jialin Pan. 2018. Data poisoning attacks on multi-task
relationship learning. In Proceedings of the 32nd AAAI Conference on Artificial Intelligence. 2628–2635.
[96] Chen Zhu, W. Ronny Huang, Hengduo Li, Gavin Taylor, Christoph Studer, and Tom Goldstein. 2019. Transferable
clean-label poisoning attacks on deep neural nets. In Proceedings of the International Conference on Machine Learning.
PMLR, 7614–7623.
Received 15 December 2021; revised 5 May 2022; accepted 13 May 2022
ACM Computing Surveys, Vol. 55, No. 7, Article 134. Publication date: December 2022.