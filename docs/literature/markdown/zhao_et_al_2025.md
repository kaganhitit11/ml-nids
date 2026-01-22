## Data Poisoning in Deep Learning: A Survey

## Pinlong Zhao, Weiyao Zhu, Pengfei Jiao, Di Gao, Ou Wu

```
Abstract—Deep learning has become a cornerstone of mod-
ern artificial intelligence, enabling transformative applications
across a wide range of domains. As the core element of deep
learning, the quality and security of training data critically
influence model performance and reliability. However, during
the training process, deep learning models face the significant
threat of data poisoning, where attackers introduce maliciously
manipulated training data to degrade model accuracy or lead
to anomalous behavior. While existing surveys provide valuable
insights into data poisoning, they generally adopt a broad
perspective, encompassing both attacks and defenses, but lack
a dedicated, in-depth analysis of poisoning attacks specifically in
deep learning. In this survey, we bridge this gap by presenting
a comprehensive and targeted review of data poisoning in deep
learning. First, this survey categorizes data poisoning attacks
across multiple perspectives, providing an in-depth analysis of
their characteristics and underlying design princinples. Second,
the discussion is extended to the emerging area of data poisoning
in large language models(LLMs). Finally, we explore critical open
challenges in the field and propose potential research directions to
advance the field further. To support further exploration, an up-
to-date repository of resources on data poisoning in deep learning
is available at https://github.com/Pinlong-Zhao/Data-Poisoning.
Index Terms—Data poisoning, deep learning, artificial intelli-
gence security.
```
### I. INTRODUCTION

# O

```
VER the past decade, machine learning, particularly
deep learning, has made remarkable progress in the field
of artificial intelligence (AI), driving transformative advance-
ments across industries and society as a whole [1]. From
image recognition [2], [3] and speech processing [4], [5] to
natural language understanding [6]–[8], deep learning models
have achieved groundbreaking success in numerous applica-
tions, significantly enhancing the automation and precision
of intelligent systems. Notably, the latest developments in
large language models (LLMs) have demonstrated exceptional
learning and reasoning capabilities [9]–[13], propelling AI
towards higher levels of intelligence and even being considered
a potential key to achieving Artificial General Intelligence
(AGI). These advancements have been primarily driven by
the availability of immense computational power and diverse,
large-scale training datasets, which together form the founda-
tion for the rapid development of modern artificial intelligence.
```
```
This work was partially supported by the National Natural Science Foun-
dation of China under Grant 62476191. (Corresponding author: Ou wu.)
Pinlong Zhao, and Weiyao Zhu contributed equally.
Pinlong Zhao and Pengfei Jiao are with the School of Cyberspace,
Hangzhou Dianzi University, Hangzhou 310018, China. E-mail: pin-
longzhao@hdu.edu.cn, pjiao@hdu.edu.cn.
Weiyao Zhu is with National Center for Applied Mathematics, Tianjin
University, Tianjin, China, 300072. E-mail: wyzhu@tju.edu.cn.
Di Gao and Ou Wu are with HIAS, University of Chinese
Academy of Sciences, Hangzhou, China, 310024. E-mail: gaodi@ucas.ac.cn,
wuou@ucas.ac.cn.
```
```
Fig. 1. Types of Adversarial Attacks.
```
```
Despite these advancements, the increasing deployment of
AI systems in critical domains such as healthcare, finance,
and transportation has raised serious concerns about their
security and reliability. The performance of AI models heavily
relies on the quality of their training data and robustness to
perturbations, yet this dependence also exposes them to a
variety of potential security threats. By crafting sophisticated
attack strategies, attackers can disrupt the normal functioning
of AI systems, leading to erroneous decisions or even catas-
trophic failures in mission-critical tasks. Among these threats,
adversarial attacks have been widely recognized as one of the
most challenging risks facing AI today. Through manipulation
of input data or exploiting model vulnerabilities, adversarial
attacks can directly undermine model performance, posing
severe threats to the safety and reliability of AI applications.
Adversarial attacks can be categorized into several types
based on their methodologies, including model attacks [14],
[15], evasion attacks (also known as adversarial exam-
ples) [16]–[18], and data poisoning attacks (also known as
data poisoning or poisoning attacks) [19]–[21], as illustrated
in Fig. 1. Model and evasion attacks typically target trained
models. Evasion attacks primarily occur during the test phase,
where adversarial examples are crafted to mislead predictions.
Model attacks, on the other hand, aim to compromise the
integrity of the model itself by tampering with its parameters,
structure, or inserting hidden functionalities. In contrast, data
poisoning attacks, which occur during the training phase,
are more covert and destructive, aiming to compromise the
training model by injecting malicious samples into the training
data. This manipulation can fundamentally degrade the model
performance or alter its behavior. Thus, comprehensively un-
derstanding and mitigating data poisoning attacks is crucial
```
## arXiv:2503.22759v1 [cs.CR] 27 Mar 2025


```
TABLE I
THECOMPARISON BETWEENOURWORK ANDEXISTINGSURVEYS
```
```
Paper Scope Data Poisoning Attacks Data Poisoning Algorithms LLMs
TML DL A D AO AG AK ASt ASc AI AV HA LP FSA BO IM GA
Pitropakis 2019 [22] ✓ ✓ ✓ ✓ ✓
Tahmasebian 2020 [23] ✓✓✓✓✓✓✓
Ahmed 2021 [24] ✓ ✓ ✓ ✓
Ramirez 2022 [25] ✓✓✓✓✓✓✓✓✓
Fan 2022 [26] ✓ ✓ ✓ ✓ ✓ ✓
Tian 2022 [27] ✓✓✓✓✓✓✓✓✓
Wang 2022 [28] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
Goldblum 2022 [29] ✓✓✓✓✓✓✓✓
Xia 2023 [30] ✓ ✓ ✓ ✓ ✓
Tayyab 2023 [31] ✓✓✓✓✓
Cina 2023 [32] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
Cina 2024 [33] ✓✓✓✓✓
Ours ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
Abbreviation: TML-Traditional Machine Learning, A-Attacks, D-Defenses; AO-Attack Objective, AG-Attack Goal, AK-Attack Knowledge, ASt-Attack Stealthiness,
ASc-Attack Scope, AI-Attack Impact, AV-Attack Variability; HA-Heuristic-based Attacks, LP-Label Flipping, FSA-Feature Space Attacks, BO-Bilevel Optimization,
IM-Influence-based Method, GA-Generative Attacks.
```
for safeguarding the security and reliability of AI systems.
In recent years, survey studies on data poisoning attacks
have steadily increased, offering a broad academic perspective
on this field [22]–[25]. However, most existing surveys adopt
a generalized approach, encompassing a wide range of topics
such as adversarial attacks and defenses across various ma-
chine learning paradigms. In particular, they often address both
traditional machine learning and deep learning technologies,
and frequently include discussions of other adversarial threats,
such as evasion and model attacks. Although these surveys
provide valuable foundational insights into data poisoning re-
search, they lack the depth and focus needed to systematically
analyze poisoning techniques tailored to deep learning models.
With the extensive deployment of LLMs in critical applica-
tions, the risks posed by data poisoning attacks in LLMs have
become increasingly concerning [34]. Unlike conventional
deep learning models, LLMs undergo multi-stage training,
including pre-training, fine-tuning, and preference alignment,
etc [35]. Each of these stages introduces potential attack
surfaces, making them uniquely susceptible to data poisoning.
Despite the widespread deployment of these models, system-
atic research on their vulnerabilities to poisoning remains
limited.
In response to these limitations, this survey focuses ex-
clusively on data poisoning attacks within the realm of deep
learning, with the goal of providing a thorough and structured
analytical framework. Specifically, this work systematically
reviews existing data poisoning methods in deep learning
and classifies these methods across multiple dimensions. By
delving into the design principles of various algorithms, this
survey seeks to offer a structured reference for advancing
research on data poisoning techniques in deep learning.
Furthermore, to ensure a concentrated and thorough analy-
sis, this survey deliberately excludes discussions on defense,
with the goal of providing a more detailed and focused explo-
ration of data poisoning algorithms themselves. Nevertheless,

```
the systematic synthesis of attack methodologies presented in
this review is expected to serve as a valuable foundation for
the future development of effective defenses.
To better highlight the distinct focus and comprehensive
coverage of our work compared to prior surveys, Table I
presents a comparative summary across several key dimen-
sions. Specifically, it illustrates how existing surveys tend to
cover a broad spectrum of adversarial attacks and defenses,
spanning traditional machine learning and deep learning al-
gorithms, but lack a detailed taxonomy of data poisoning
techniques and algorithms. Furthermore, none of the existing
surveys have systematically explored data poisoning in LLMs,
despite their growing prominence in critical AI applications.
In contrast, our survey offers an in-depth and systematic clas-
sification of data poisoning attacks, algorithms, and extends
the discussion to LLMs, analyzing the unique attack vectors
introduced by their multi-stage training paradigm.
The main contributions of this survey can be summarized
as follows:
1) Comprehensive Understanding. This survey provides an
in-depth and comprehensive overview of data poisoning in
deep learning, offering a detailed theoretical framework that
helps researchers better understand the mechanisms, implica-
tions, and challenges of these attacks. By focusing exclusively
on data poisoning attacks, this work indirectly supports the
development of more robust defense mechanisms through a
thorough analysis of attack strategies.
2) Taxonomy of Data Poisoning Attacks. We present a
systematic taxonomy of data poisoning attacks, constructed
along two key dimensions to capture both conceptual and
technical perspectives. The first dimension focuses on clas-
sifying attacks based on their characteristics, while the second
categorizes data poisoning attacks according to the funda-
mental principles underlying their algorithms. By integrating
these two perspectives, this survey offers a comprehensive
and structured taxonomy that facilitates a clear understanding
```

Fig. 2. Deep learning and data poisoning attack pipeline.

of data poisoning attacks and serves as a practical guide for
researchers exploring this evolving field.
3) Data Poisoning in Large Language Models (LLMs).
To our knowledge, this is the first survey that systemati-
cally explores data poisoning attacks in LLMs. Given the
increasing deployment of LLMs in critical applications and
their unique vulnerabilities across various training stages, this
work provides a systematic and in-depth examination of how
data poisoning can compromise these powerful models. By
highlighting vulnerabilities across different training stages, we
aim to fill a critical gap in the current literature.

4) Future Research Directions. We discuss some possible di-
rections that could shape the future of data poisoning research.
These include enhancing the stealth and effectiveness of poi-
soning attacks, exploring vulnerabilities in dynamic learning
environments such as federated and continual learning, and ex-
panding the study of data poisoning to multimodal AI systems
and emerging architectures. Furthermore, we emphasize the
need for standardized benchmarks and evaluation frameworks
to ensure consistent and rigorous assessment of poisoning
techniques.

5) Online Resources and Repository. To facilitate further
research, this survey provides an open-source online repository
containing a curated collection of relevant works on data
poisoning attacks, including links to papers, datasets, and
codes. This resource offers a convenient tool for researchers
to track the latest developments and explore state-of-the-
art techniques. By continuously updating the repository, this
survey ensures it remains a valuable and evolving reference
for the community.

The rest of this paper is organized as follows: In Section
II, we introduce the fundamental concepts of data poisoning,
laying the groundwork for understanding its key principles and
implications. In Section III, we present the taxonomy of data
poisoning attacks. In Section IV, we delve into the taxonomy
of data poisoning algorithms, emphasizing the basic principles

```
of the algorithms. Section V explores data poisoning in the
context of LLMs. In Section VI, we propose potential research
directions to advance the field further. Finally, we conclude
this survey in Section VII.
```
```
II. PRELIMINARY
In this section, we provide a foundational overview of data
poisoning in the context of deep learning. First, we discuss
security attacks in deep learning, focusing on vulnerabilities
introduced during the training phase. Next, we briefly outline
the deep learning training pipeline and the fundamental con-
cepts of data poisoning to aid in understanding and analyzing
the subsequent sections of this survey. Finally, we introduce
the characteristics posed by data poisoning in deep learning.
```
```
A. Security Attacks in Deep Learning
Deep learning is a cornerstone of modern artificial intel-
ligence, enabling breakthroughs in computer vision, natural
language processing, and autonomous systems [36]. Its success
relies on large, high-quality datasets for training sophisticated
models. However, as these models are increasingly deployed
in high-stakes applications such as healthcare, finance, and
autonomous vehicles, ensuring their reliability and robustness
has become a growing concern [37]–[39].
One of the most critical vulnerabilities lies in the reliance
on large-scale datasets, often aggregated from diverse and
potentially untrusted sources [40]. This dependency creates
opportunities for attackers to exploit the training phase, al-
lowing attackers to inject malicious data that undermines
the model integrity before deployment. Unlike adversarial
examples, which perturb inputs at the inference stage, data
poisoning attacks target the training process, embedding sys-
tematic biases or backdoors that persist across deployments,
compromising model reliability. These risks highlight the
importance of understanding and mitigating data poisoning
attacks in deep learning [41].
```

B. Data Poisoning Attack Pipeline

Deep learning models are typically trained in a pipeline
involving data collection, preprocessing, model training, and
evaluation, as illustrated in Fig. 2. Data poisoning refers to
the deliberate injection of carefully crafted malicious samples
into the training dataset of a machine learning model, with
the intent of manipulating its behavior. By altering the data
distribution during training, attackers can achieve a variety of
objectives, ranging from degrading overall model performance
to implanting specific vulnerabilities that can be exploited
post-deployment.
The general framework for a data poisoning attack involves
the following stages:

- Data Collection.Attackers exploit vulnerabilities in the
    data acquisition pipeline, such as reliance on open-
    source datasets, web scraping, or third-party data vendors.
    Poisoned data can be introduced at this stage without
    immediate detection.
- Data Poisoning.Poisoned samples are carefully crafted
    to blend with benign data and evade anomaly detection
    mechanisms. Attackers may modify existing samples or
    insert synthetic data to manipulate the training distribu-
    tion.
- Model Contamination.The poisoned dataset skews the
    learning process of the model, causing it to associate irrel-
    evant or deceptive features with target labels, ultimately
    distorting decision boundaries.
- Evaluation and Deployment.Once deployed, the com-
    promised model exhibits vulnerabilities, such as targeted
    misclassifications or specific responses to hidden triggers.

C. Characteristics of Data Poisoning in Deep Learning

Data poisoning attacks in deep learning differ significantly
from traditional machine learning poisoning attacks due to
the extensive data volumes and high model complexity. The
characteristics of deep learning amplify the impact and com-
plexity of data poisoning attacks. The following are the unique
characteristics of data poisoning attacks in deep learning:
The general framework for data poisoning attacks involve
the following stages:

- Dependence on Large-Scale Data.Deep learning models
    rely on massive datasets collected from diverse and often
    unverified sources. The sheer volume and heterogeneity
    of the data make it impractical to thoroughly inspect
    and validate every sample, increasing the likelihood of
    introducing poisoned data.
- High Model Complexity. Deep neural networks have
    substantial capacity and are capable of memorizing out-
    liers or poisoned samples without significantly degrading
    performance on benign samples. This ability to generalize
    while retaining specific patterns makes it easier for attack-
    ers to embed malicious behavior that remains dormant
    until activated under specific conditions [42].
- Distributed Training Environments.Decentralized train-
    ing paradigms, such as federated learning, involve mul-
    tiple participants contributing data and computational re-
    sources. Malicious actors within these environments can

```
inject poisoned data without direct access to the global
model, complicating detection and accountability [43]–
[45].
```
- Evaluation Difficulties.The nonlinearity and high di-
    mensionality of deep learning models make it difficult
    to assess the full impact of poisoning attacks. Existing
    evaluation metrics and tools are often insufficient for
    quantifying the subtle yet significant effects of such
    attacks [46].

### III. TAXONOMY OF POISONING ATTACKS

```
Poisoning attacks were first introduced by Barreno et
al. [47] and have since emerged as a major security threat
during the training phase of machine learning systems, particu-
larly in the context of deep learning [48]. These attacks exploit
vulnerabilities in the training process by manipulating data,
thereby compromising the integrity, availability, or reliability
of the resulting model [49]. Over time, poisoning attacks
have garnered significant attention due to their potential to
undermine critical applications of deep learning [50], [51].
In this section, we systematically categorize poisoning at-
tacks by reviewing representative studies, as illustrated in
Fig. 3. The taxonomy is constructed along seven distinct
dimensions, focusing on the characteristics of the attacks,
including their objectives, goals, attacker knowledge, and other
relevant factors. This taxonomy provides a structured frame-
work for systematically analyzing and comparing the various
characteristics of data poisoning attacks. By categorizing these
attacks, this section aims to clarify their defining charac-
teristics, differentiate attack types, and lay the groundwork
for further research and the development of effective defense
strategies.
```
```
A. Attack Objective
Data poisoning attacks can be categorized based on their
objectives into three main types: label attack (label mod-
ification or label flipping attack) [52], input attack (input
modification or clean-label attack) [29], and data attack (data
modification) [28]. Each type targets different manipulated-
target of the training data, affecting the performance of deep
learning models.
Label Modification Attacks.Label modification attacks aim
to disrupt model training by altering the labels of selected
samples while keeping the input features unchanged [53]–
[57]. A typical example is the label-flipping attack, where the
labels of specific samples are deliberately changed to incorrect
classes, distorting the model’s decision boundary. Zhao et
al. [53], proposed a bilevel optimization-based label manipula-
tion method, solved using a Projected Gradient Ascent (PGA)
algorithm, which is both effective and transferable to black-
box models. Jha et al. [56] proposed the FLIP framework,
which leverages trajectory matching to perform label-only
backdoor attacks.
Input Modification Attacks.Input modification attacks, also
known as clean label attacks, involve perturbing the input
features of training samples while preserving their original
labels. Feature collision, a type of clean-label attack, generates
```

Fig. 3. Taxonomy of data poisoning attacks in deep learning with representative examples.

adversarial examples aligned with target samples in the feature
space. This alignment causes the model to associate these
examples with the target class during inference [58]–[62].
Bilevel poisoning can also retain the original labels. It employs
a bilevel optimization framework to create perturbed inputs,
which degrade the model’s performance during training [63]–
[66].

Data Modification Attacks.Data modification attacks com-
bine alterations to both input features and labels, or creating
entirely fabricated samples to achieve the attacker’s goals.
Most backdoor attacks fall under this category, where trigger
patterns are embedded in the data to cause the model to
exhibit specific behavior under predefined conditions [67]–
[70]. Generative methods, such as attacks utilizing genera-
tive adversarial networks (GANs), can produce high-quality

```
synthetic samples that are indistinguishable from legitimate
data, effectively poisoning the training process while evading
detection [71]–[73]. By enabling simultaneous manipulation
of input features and labels, data modification attacks offer
greater potential to execute sophisticated and highly targeted
attacks, significantly undermining the model’s security and
reliability.
```
```
B. Attack Goal
The goal of data poisoning attacks is to manipulate the
training data to compromise the performance or integrity of
machine learning models. Based on the adversarial intent,
these attacks are categorized into untargeted attack (indiscrim-
inate attack or availability attack), targeted attack (integrity
```

attacks), and backdoor attacks (integrity attacks). Below, we
discuss these attack types along with relevant studies.
Untargeted Attacks.Untargeted attacks seek to degrade the
overall performance of a machine learning system, reducing
its usability for legitimate tasks. These attacks often involve
injecting corrupted samples or perturbing existing data to
cause widespread misclassification [66], [74]–[77]. Yang et
al. [75] propose a generative framework for poisoning attacks
on neural networks, using an autoencoder to efficiently create
poisoned samples. Reference [76] describes an approach where
the attacker utilizes a generative model to produce clean-label
poisoning samples designed to undermine the victim model.
Fowl et al. [66] employed a gradient alignment optimization
method to subtly modify the training data, effectively reducing
the model’s performance. These studies illustrate the variety of
techniques attackers can leverage to compromise the reliability
and overall functionality of deep learning models.
Targeted Attacks.Targeted attacks aim to mislead the model
on specific target samples while preserving its general perfor-
mance on clean data. Such targeted attacks are particularly
insidious, as they ensure the system appears functional to
most users [58], [60], [61], [78], [79]. Shafahi et al. [58]
proposed a poisoning framework for deep learning models that
generates subtle perturbations to achieve targeted misclassifi-
cation. Jagielski et al. [79] proposed a subpopulation attack
that targets the performance of a machine learning model on
a specific subpopulation, while preserving its accuracy on the
remaining dataset.
Backdoor Attacks.Backdoor attacks involve embedding a
trigger pattern into the training data, enabling the adversary
to manipulate model predictions when the trigger is present,
while maintaining high accuracy on clean samples [67], [80]–
[83]. Gu et al. [67] introduced “BadNets”, which demonstrated
that injecting a small number of poisoned samples with an
embedded trigger into the training data can effectively implant
a backdoor into the model. Chen et al. [80] advanced backdoor
attacks by introducing imperceptible noise patterns as triggers,
making detection even more challenging. Another variation,
proposed by Turner et al. [81], utilizes label-consistent back-
door triggers to minimize visual discrepancies between poi-
soned and clean samples, further evading detection. Backdoor
attacks demonstrate the severe risks posed by data poisoning
in outsourced or collaborative deep learning workflows.

C. Attack Knowledge

The attacker’s knowledge plays a critical role in deter-
mining the feasibility and effectiveness of data poisoning
attacks. Based on the level of access to the victim’s machine
learning system, attacks can be categorized into three types:
Black-box attacks, Grey-box attacks, and White-box attacks.
This classification reflects the attacker’s access to information
about the model architecture, parameters, and training data.
Understanding these categories is crucial for evaluating the
feasibility and impact of different attack strategies in practical
scenarios.
White-box attacks.In this setting, the adversary has full
knowledge of the victim model, including its architecture,

```
parameters, and the training dataset. This extensive knowl-
edge allows the adversary to craft highly precise poisoning
samples that maximize the attack’s effectiveness. Techniques
such as bilevel optimization are often employed in white-box
scenarios to identify and inject poisoned samples that directly
manipulate the model’s decision boundaries. While white-box
attacks are not always feasible in real-world applications due
to the difficulty of obtaining such privileged access, they serve
as an important benchmark for evaluating a worst-case [58],
[76], [84], [85].
Black-box Attacks.In this case, the attacker has no direct
access to the victim model’s internal details, such as its ar-
chitecture, parameters, or training data. Instead, the adversary
interacts with the system through its input-output behavior,
typically by querying the model and observing its predictions.
These attacks rely on techniques such as transfer learning,
where a surrogate model is used to approximate the target
model’s behavior. Black-box poisoning attacks are particularly
challenging to detect, as they exploit limited information
while still effectively degrading the model’s performance [86]–
[88]. For instance, black-box attacks on recommender systems
manipulate user-item interaction data without requiring access
to the underlying recommendation algorithm [89], [90].
Grey-box Attacks.The attacker has partial knowledge of
the victim model, such as the architecture or a subset of
the training data, but lacks full access to critical details
like the complete training set or exact parameter values.
This intermediate level of knowledge enables the attacker to
perform more targeted and efficient poisoning compared to
black-box scenarios [14], [91]–[93]. For example, an adversary
with access to a pre-trained model but not the fine-tuning data
may craft poisoning samples to disrupt the fine-tuning process.
Grey-box attacks highlight the risks associated with shared or
collaborative training environments, where some information
about the model may be inadvertently exposed.
```
```
D. Attack Stealthiness
Based on the detectability of adversarial modifications in
the training data, poisoning attacks can be classified into non-
stealthy attack and stealthy attack.
Non-stealthy Attacks.Non-stealthy attacks involve notice-
able modifications to the training data, often injecting con-
spicuous anomalies or significantly altering existing sam-
ples. These attacks prioritize maximizing their impact on
the model’s performance rather than evading detection. For
instance, in [67], [68], [94], the authors introduced trojaning
attacks on neural networks, where an attacker embeds mali-
cious behavior by reverse-engineering the model to generate
a trigger and retraining it with synthetic data. The trigger
activates specific neurons, causing the model to produce ad-
versarial outputs while maintaining normal performance on
benign inputs.
Stealthy Attacks.In contrast to non-stealthy attacks, stealthy
attacks leverage subtle modifications to the training data,
ensuring the poisoned samples remain statistically similar
to clean data and thus evade detection. These attacks are
particularly challenging to defend against as they carefully
```

balance perturbation magnitude and attack efficacy [59], [95]–
[98]. Suciu et aly [59] introduced the ”StingRay” attack, which
leverages targeted poisoning to alter the decision boundary
of machine learning models. The StingRay attack generates
poisoned samples that are highly similar to clean training
instances in the feature space. Li et al. [96] introduced
two novel techniques for embedding triggers into models:
steganography and regularization. These methods effectively
ensure the success of the attack while preserving the original
functionality of the model and achieving a high level of
stealth. Xu et al. [97] introduced “Shadowcast,” a stealthy data
poisoning attack specifically targeting vision-language mod-
els (VLMs). Shadowcast constructs visually indistinguishable
poisoned image-text pairs, manipulating the model to generate
adversarial outputs while maintaining coherence and subtlety.

E. Attack Scope

Poisoning attacks can be categorized by the breadth of their
impact on the target model, ranging from narrowly focused
attacks on individual samples to broader attacks that disrupt
multiple categories or entire datasets. These classifications are
useful for understanding the scalability and specificity of the
threat posed by poisoning attacks.
Single-instance Attacks.Single-instance attacks focus on
inducing misclassification of a specific target sample without
affecting the overall model performance [58], [59], [80]. For
instance, the feature collision method introduced by Shafahi
et al. generates adversarial examples in the training set that
cause a specific test instance to be misclassified into a desired
class [58]. These attacks often require precise feature manip-
ulation and are commonly applied in clean-label settings to
remain undetectable.
Single-pattern attacks.Single-pattern attacks aim to mis-
classify a group of inputs sharing a specific pattern. The back-
door attack strategy is a prominent example, where attackers
introduce a trigger pattern during training that reliably acti-
vates a specific malicious behavior at inference [67], [72], [73],
[99]. These attacks demonstrate high stealth and flexibility, as
the triggers are often imperceptible yet effective in subverting
the model.
Single-class Attacks.In single-class attacks, the objective
is to disrupt the classification of all instances belonging to
a particular class while preserving the performance of the
model on other classes. Methods such as bilevel optimization
have been shown to craft poison samples that compromise
an entire class with minimal perturbations [100]–[102]. This
attack type is particularly concerning in scenarios involving
sensitive classifications, such as biometric systems or medical
diagnostics.
Broad-scope Attacks.Broad-scope attacks seek to degrade
the performance of the model across multiple classes or the
entire dataset. These attacks are often employed in availability
attacks, where the goal is to render the model unusable or
significantly degrade its general performance [75]–[77]. For
instance, attackers might inject a large number of poisoned
samples into the training set, affecting the model’s overall ac-
curacy and reliability, thus undermining its ability to function
correctly in a real-world environment.

```
F. Attack Impact
Poisoning attacks can be categorized according to the spe-
cific impact they are designed to achieve, including perfor-
mance attack, robustness attack, and fairness attack. Each cate-
gory addresses a different dimension of vulnerability exploited
by attackers.
Performance Attacks.The primarily of this type of attack
is to degrade the overall accuracy or usability of the model
by introducing poisoned samples into the training dataset.
These attacks often exploit weaknesses in the training process
to induce widespread misclassification. For example, methods
such as bilevel optimization and reinforcement learning have
been used to craft poisoning samples that disrupt the overall
performance of models across tasks like image classification
systems [64], [65], [103], [104].
Robustness Attacks. Instead of targeting overall perfor-
mance, robustness attacks focus on undermining the resilience
of the model to perturbations or adversarial examples, making
it more vulnerable to malicious inputs during deployment.
Zheng et al. [105] proposed a concealed poisoning attack that
reduces the robustness of deep neural networks by generating
poisoned samples through a bi-level optimization framework.
This approach not only degrades the model’s resistance to
adversarial inputs but also ensures high stealth by maintaining
performance on clean samples. Similarly, Alahmed et al. [106]
investigated the impact of poisoning attacks on deep learning-
based network intrusion detection systems. [107] proposes an
Adversarial Robustness Poisoning Scheme (ARPS) that subtly
degrades a model’s adversarial robustness while preserving its
normal performance, posing a stealthy threat to deep neural
networks.
Fairness Attacks.Fairness attacks target the ethical and equi-
table functioning of the model by skewing its decisions toward
biased outcomes. These attacks exploit vulnerabilities in the
training process to introduce systemic biases. For instance,
solans et al. [108] proposed a gradient-based poisoning attack
designed to increase demographic disparities among groups by
introducing classification inequities. This approach effectively
manipulates model behavior in both white-box and black-
box scenarios, showcasing the adaptability of such attacks
across different settings. Similarly, Van et al. [109] devel-
oped a framework for generating poisoning samples through
adversarial sampling, labeling, and feature modification. This
framework enables attackers to adjust their focus on fairness
violation or accuracy degradation, demonstrating significant
impacts on group-based fairness notions such as demographic
parity and equalized odds. Furth et al. [110] introduced the
“Un-Fair Trojan” attack, a backdoor approach that targets
model fairness while remaining highly stealthy. This method
uses a trojan trigger to disrupt the fairness metrics of the
model, significantly increasing demographic parity violations
without reducing overall accuracy.
```
```
G. Attack Variability
The persistence and adaptability of data poisoning attacks
can vary significantly depending on how the adversary in-
jects and maintains poisoned data within the system. While
```

some attacks rely on fixed, predefined manipulations, others
continuously evolve, making detection and mitigation more
challenging. This distinction leads to two primary categories:
static attacks and dynamic attacks.

Static Attacks. Static attacks involve injecting poisoned
data into the training set before the model is trained, with
the adversarial modifications remaining unchanged throughout
the entire lifecycle of the model. These attacks often use
fixed persistent poisoning strategies that do not adapt post-
deployment. Most data poisoning attacks, including label-
flipping and fixed-pattern backdoor attacks, fall under this
category [81], [111]–[113]. A well-known example of static
poisoning is BadNets [67], which embeds a fixed trigger into
training samples, causing the model to consistently misclassify
inputs containing the trigger at inference time. While static
attacks can be highly effective, their reliance on fixed patterns
makes them more susceptible to detection by anomaly detec-
tion and robust training techniques.

Dynamic attacks.Dynamic attacks introduce adaptive poi-
soning strategies, where the attack evolves over time to en-
hance stealth and robustness. Salem et al. [114] introduced
dynamic backdoor attacks, such as Backdoor Generating Net-
work (BaN) and Conditional BaN (c-BaN), which generate
variable triggers across different inputs. These adaptive trig-
gers allow backdoored models to evade traditional defenses
like Neural Cleanse and STRIP. Similarly, Nguyen et al. [71]
proposed input-aware backdoor attacks, where triggers are
generated conditionally based on the input data, making detec-
tion significantly more challenging. Beyond backdoor attacks,
PoisonRec by Song et al. [89] presents an adaptive poisoning
framework for black-box recommender systems, using rein-
forcement learning to iteratively refine poisoning strategies
based on system feedback. This method demonstrates that
adaptive poisoning can extend beyond classification models to
real-world recommender systems, increasing the persistence
and effectiveness of attacks.

### IV. DATA POISONING ALGORITHMS

The effectiveness of data poisoning attacks is determined by
the strategies used to manipulate training data. These strategies
differ in terms of computational complexity, the extent of
adversarial control, and their ability to evade detection. A
precise understanding of these techniques is crucial for both
the development of new poisoning methods and the design of
effective defenses.
This section examines the fundamental principles and math-
ematical foundations of poisoning algorithms, focusing on how
they alter training data, optimize attack objectives, and exploit
model vulnerabilities. Fig. 4 illustrates six major algorithmic
approaches used in data poisoning: heuristic-based attacks,
label flipping, feature collision attacks, bilevel optimization,
influence-based methods, and generative model-based attacks.
To provide a comprehensive overview, Table II presents a sum-
mary of data poisoning algorithms. The following subsections
provide a detailed analysis of these methods:

```
A. Heuristic-based Attacks
Heuristic-based attacks represent one of the fundamental
approaches to data poisoning in deep learning. These attacks
leverage predefined heuristic rules rather than complex opti-
mization frameworks to craft poisoned data samples. Unlike
sophisticated bilevel optimization or influence-based meth-
ods, heuristic-based approaches typically rely on empirical
observations and domain knowledge to manipulate training
data. While they are often less effective in achieving highly
targeted attacks, their simplicity and efficiency make them
widely applicable, especially in scenarios where computational
constraints exist.
A foundational heuristic-based attack is BadNets [67],
which introduces a straightforward poisoning strategy: a small
trigger pattern is injected into selected training samples, and
their labels are modified to a target class. This results in a
trained model that behaves normally on clean inputs but mis-
classifies any input containing the trigger. However, BadNets-
style attacks often introduce detectable data anomalies, making
them susceptible to defensive techniques such as anomaly
detection and adversarial training.
To enhance stealth, subsequent research introduced more
sophisticated injection techniques. Chen et al [80] demon-
strated that only a small number of poisoned samples without
direct access to the model are sufficient to implant an effective
backdoor. Similarly, Alberti et al. [100] explored a minimal
perturbation approach, showing that modifying just a single
pixel per image across the dataset can establish a functional
backdoor. This attack emphasizes how seemingly negligible
changes can have disproportionate effects on deep learning
models. An even more covert strategy was proposed in [96],
which improves upon BadNets by embedding triggers in a
way that remains imperceptible to both human inspectors
and anomaly detection algorithms. This is achieved through
steganographic techniques, ensuring that backdoors remain
hidden while maintaining a high attack success rate. Pix-
door [115] created a nearly undetectable backdoor attack with
minimal poisoned data injection by manipulating only the least
significant bits of pixel values.
```
```
B. Label Flipping Attacks
High-quality training data is essential for ensuring the
accuracy and reliability of machine learning models. Since
these models learn by associating input data with correct
labels, any disruption in this relationship weakens their per-
formance. Label flipping attacks exploit this vulnerability by
selectively altering training labels while leaving the actual
data unchanged. While these attacks do not maintain clean
labels, they avoid introducing visible artifacts that could make
manipulation obvious. By embedding false associations into
the learning process, these attacks mislead the model, causing
systematic errors or targeted misclassification. A label flipping
attack can be formally expressed as follows:
```
```
LF(y) =
```
### 

```
1 −y, y∈Y={ 0 , 1 }
random(Y\{y}), others
```
### (1)

```
whereyrepresents the original label, and 1 −ydenotes the
flipped label in a 0/1 classification task. For multi-class clas-
```

```
TABLE II
SUMMARY OF DATA POISONING ALGORITHMS
```
```
Algorithms Attacks Ref. Year Model Application
```
Heuristic-based Attacks BadNets [67] 2017 CNN, Faster-RCNN

```
handwritten digit
recognition, traffic sign
detection
Chen et al. [80] 2017 DeepID, VGG-Face face recognition, face
verification
Alberti et al. [100] 2018
lexNet, VGG-16,
ResNet-18, DenseNet-121 image classification
Li et al. [73] 2021 ResNet-18 image classification
Pixdoor [115] 2021 LeNet handwritten digitrecognition
Label Flipping Attacks Zhang et al. [116] 2021 AlexNet, Inception V3 image classification
Zhang et al. [117] 2021 AlexNet, LeNet spam classification
Li et al. [55] 2022 MLP malware detection
```
```
FLIP [56] 2023
```
```
ResNet-32, ResNet-18,
VGG-19, Vision
Transformer
```
```
image classification
```
```
Lingam et al. [118] 2024
GCN, GAT, APPNP,
CPGCN, RTGNN node classification.
Feature Space Attacks Poison Frogs [58] 2018 InceptionV3, AlexNet image classification
```
```
FAIL [59] 2018 NN
```
```
malware detection, image
classification, exploit
prediction, data breach
prediction
Convex Polytope [60] 2019
ResNet-18, ResNet-50,
DenseNet-121, SENet-18 image classification
Saha et al. [99] 2020 RAlexNet image classification
BlackCard [61] 2020 ResNet, DenseNet image classification, facerecognition
```
```
Bullseye
Polytope [62]^2021
```
```
SENet18, ResNet50,
ResNeXt29-2x64d,
DPN92, MobileNetV2,
GoogLeNet
```
```
image classification
```
```
Luo et al. [119] 2022 ResNet18 image classification
Bilevel Optimization
Attacks
```
```
Munoz-Gonz ̃ ́alez
et al.
[74] 2017 CNN
```
```
spam filtering, malware
detection, handwritten
digit recognition
```
MetaPoison [65] (^2020) VGG13,ResNet20ConvNetBN, image classification
Witches’ Brew [64] 2020 ConvNet, ResNet-18 image classification
Pourkeshavarz et
al.
[120] 2024
PGP, LaPred, HiVT,
TNT, MMTransformer,
LaneGCN
trajectory prediction
BLTO [121] 2024 SimCLR, BYOL,
SimSiam
feature extractor
Influence-based Attacks Koh and Liang [122] 2017 CNN
understanding model
behavior, debugging
models, detecting dataset
errors
Basu et al. [123] 2021 small CNN, LeNet ,
ResNets, VGGNets
image classification
Koh et al. [85] 2022 neural network spam detection, sentimentclassification
Generative Attacks Yang et al. [75] 2017 auto-encoder image classification
Feng et al. [76] 2019 auto-encoder image classification
Munoz-Gonz ̃ ́alez
et al. [77]^2019 pGAN image classification
Psychogyios et
al.
[124] 2023 GAN grapevine image
classificationn
Chen et al. [125] 2024 GAN cloud API recommender


Fig. 4. The sub-taxonomy of data poisoning algorithms.

sification, the functionrandom(·)selects samples randomly
for label flipping.
In addition to randomly selecting, an attacker can strategi-
cally choose a subset of data for flipping to achieve a stronger
attack impact. To understand how label flipping affects deep
learning models, Zhang et al. [116] conducted experiments
to evaluate the impact of random label corruption on neural
networks. Their study revealed that deep models can perfectly
fit training data even when labels are completely randomized,
achieving zero training error while failing to generalize on test
data. This overfitting behavior demonstrates that deep learning
models do not inherently distinguish between correct and in-
correct labels, making them highly susceptible to label flipping
attacks. Furthermore, standard regularization techniques such
as weight decay and dropout were found to be ineffective
in mitigating the effects of mislabeled data. Since models
trained on flipped labels internalize incorrect associations,
their decision boundaries become distorted, leading to long-
term degradation in performance. While victims may detect
label flipping through abnormal test errors, by the time this
occurs, significant computational resources have already been
expended, making it an effective denial-of-service attack.
Label flipping attacks have also been explored in different
domains and model architectures, revealing their wide-ranging
impact. Zhang et al. [117] demonstrated that label flipping is
particularly effective in spam filtering systems, where flipping
a small fraction of spam labels to legitimate ones significantly
degrades detection accuracy. This highlights the broader risk of
label flipping in security-sensitive applications where models
rely on clean labels for decision-making. Further, Li et al. [55]
proposed a targeted label flipping strategy using clustering

```
techniques to identify and alter the most influential samples.
Their method showed that strategically flipping a small subset
of labels can be far more damaging than random flipping,
making detection and mitigation more challenging.
Beyond degrading overall model performance, label flipping
can also be used for backdoor injection. Jha et al. [56] intro-
duced FLIP, a label-only backdoor attack demonstrating that
attackers can implant backdoors in models without modifying
input features. Their study revealed that even corrupting just
2% of labels in datasets like CIFAR-10 could achieve near-
perfect backdoor success rates, posing a major risk in crowd-
sourced training environments. This vulnerability is even more
pronounced in graph neural networks (GNNs), as shown by
Lingam et al. [118], who found that flipping even a single label
in graph-based models can significantly disrupt classification
accuracy due to the structural dependencies between data
points.
```
```
C. Feature Space Attacks
Feature space attacks manipulate training data so that the
feature representations of poisoned samples become indistin-
guishable from a specific target. Unlike label flipping poison-
ing attacks that alter labels, this method focuses on feature
space manipulation, ensuring that poisoned samples appear
natural while influencing the decision boundary of the model.
By injecting carefully crafted samples into the training set, the
attacker can make a chosen target sample misclassified without
modifying the target sample itself. Feature space poisoning
attacks offer three key stealth advantages. First, modifying
feature space associations does not require altering labels,
making the attack highly inconspicuous. Second, the attack
```

can introduce only minor perturbations to input samples using
optimization-based techniques, without embedding noticeable
poisoning patterns, thereby evading manual inspection. Third,
these attacks typically affect only the classification of specific
target samples while leaving non-target samples unaffected,
making detection significantly more challenging.
The first feature space attack was processed by Shafahi et al.
in 2018 [58], called feature collision attack. This attack focuses
on misclassifying specific target samples while keeping the
poisoned samples visually inconspicuous. This attack modifies
the deep features of certain training data, making them closer
to the target class in feature space, thereby misleading the
model into misclassifying the target sample during inference.
Its optimization objective includes maximizing the feature
similarity between poisoned samples and the target class while
maintaining visual similarity to the original class to enhance
stealth. The optimization objective of feature collision attack
is defined as follows:

```
xp= arg min
xp
∥f(xp)−f(xt)∥^22 +β∥xp−xb∥^22 (2)
```
wherexpis the poisoned sample,xtis the target test sample,
xb is a base class sample in the training data, f is the
target model,f(·)is the output of the model, andβis a
hyperparameter. In Equation (2), the first term makes the
poisoned sample close to the attack target categoryt, achieving
the attack purpose. The second term controls the poisoned data
to be similar to the base class data, so that there is no obvious
visual difference between the two. This attack is primarily used
in transfer learning, especially in scenarios where pre-trained
models are fine-tuned. Poisoning only a small number of sam-
ples can effectively manipulate classification results. However,
the method relies on the attacker having full knowledge of the
targeted model, which limits its practicality. Additionally, if
the victim model is later retrained with new clean data using
end-to-end training or layer-wise fine-tuning, the effects of the
poisoned data will gradually wear off.
While the feature collision attack is effective, it requires
white-box access to the victim model. To overcome this lim-
itation, Suciu et al. [59] proposed the FAIL adversary model
and introduced StingRay, a clean-label poisoning method
that generalizes across multiple neural network architectures.
StingRay avoids precise feature-space manipulation, improv-
ing attack transferability and stealth, making it more practical
and broadly applicable.
Building on these findings, Zhu et al. [60] proposed the
Convex Polytope Attack, which enhances transferability by
placing multiple poisoned samples around the target in feature
space, increasing misclassification across different classifiers.
Its optimization objective is formulated as:

```
min
{c(ji)},{x(pj)}
```
### 1

```
2 m
```
```
Xm
```
```
i=
```
```
f(i)(xt)−
```
```
Pk
j=1c
```
```
(i)
j f
(i)(x(pj))
2
```
(^) f(i)(xt)

s.t.
Xk
j=
c(ji)= 1, c(ji)≥ 0 , ∀i,j;
x(pj)−x(bj)
∞
≤ε, ∀j.

### (3)

```
where xp represents the poisoned sample, xt denotes the
target test sample, andxbis a base-class sample from the
training data. A set of pretrained models is defined asf(i)(xt),
withmrepresenting the number of sets in the model. The
attack constructskpoisoned samples, which collectively form
an enclosing structure around the target in feature space. A
constraint
```
```
Pk
j=1c
```
```
(i)
j = 1,c
```
```
(i)
j ≥^0 ensures that the weights
assigned to these enclosing poisoned samples are all positive
and sum to one. Additionally, the upper bound of perturbation
is defined asε, controlling the maximum modification applied
to each poisoned sample. The experiments showed that Convex
Polytope significantly outperforms standard feature collision
attacks in black-box settings, achieving a 50% attack success
rate while poisoning only 1% of the training data.
Despite its strong transferability, the Convex Polytope At-
tack suffered from high computational costs. To address this,
Bullseye Polytope [62] optimized the attack by centering the
target within the poisoned polytope, improving stability, effi-
ciency, and attack success rates. Meanwhile, BlackCard [61]
removed the need for model knowledge by crafting universal
poisoned samples that generalize across architectures, signif-
icantly enhancing stealth. Feature-space poisoning has also
been adapted for backdoor attacks, as seen in Hidden Trigger
Backdoor Attacks [99], which conceal triggers during training,
and Luo et al. [119], who introduced image-specific triggers
to further evade detection.
```
```
D. Bilevel Optimization Attacks
Bilevel optimization formalizes data poisoning attacks as a
two-level problem: Inner optimization: the victim model is
trained on a dataset that includes poisoned samples. Outer
optimization: the attacker optimizes the poisoned data to
maximize the desired attack effect, such as misclassification or
performance degradation. Mathematically, bilevel optimization
poisoning can be expressed as:
D′p= arg max
Dp
```
```
F(Dp,θ′) =Lout(Dval,θ′)
```
```
s.t. θ′= arg min
θ
Lin(D∪Dp,θ)
```
### (4)

```
whereD,Dval, andDprepresent the original training dataset,
the validation dataset, and the poisoned dataset, respectively.
The inner and outer loss functions are denoted asLin and
Lout. The objective of the outer optimization is to generate
a poisoned dataset that maximizes the classification error
of the target modelθ′on the clean validation datasetDval.
Meanwhile, the inner optimization iteratively updates the
target model using the poisoned datasetD∪Dp, ensuring
that the model is trained on compromised data. Since the
model parametersθ′are implicitly determined by the poisoned
datasetDp, the functionF is introduced to represent the
dependency betweenθ′andDpin the outer optimization step.
The bilevel optimization process operates as follows: once
the inner optimization reaches a local minimum, the outer
optimization updates the poisoned datasetDpusing the newly
trained target modelθ′. This process repeats until the outer
loss functionLout(Dval,θ′)converges.
The above bilevel optimization attack is an indiscriminate
attack, as its primary objective is to maximize the overall
```

classification error of the target model. However, bilevel opti-
mization can be adapted for other attack types, such as targeted
attacks and backdoor attacks. In the case of a targeted attack,
the bilevel problem transforms into a min-min optimization
problem, formally defined as follows:

```
D′p= arg max
Dp
F(Dp,θ′) =Lout({xt,yadv},θ′)
```
```
s.t. θ′= arg min
θ
Lin(D∪Dp,θ)
```
### (5)

here, yadv is the incorrect target class predefined by the
attacker. In this case, the objective of the outer optimization is
to generate a poisoned dataset that minimizes the classification
error of the target modelθ′on the designated target samples,
ensuring that they are misclassified intoyadv. Gradient-based
approaches approximate this optimization iteratively, using the
chain rule to calculate gradients ifLoutis differentiable:

```
∇DpF=∇DpLout+
```
```
∂θ
∂Dp
```
```
⊤
∇θLout
```
```
s.t.
```
```
∂θ
∂Dp
```
```
⊤
= (∇Dp∇θLin)(∇^2 θL 2 )−^1
```
### (6)

where ∇DpF represents the partial derivative of F with

respect toDp. The poisoned dataDp(i) of the i-th iteration
can be updated toD(pi+1)by gradient ascent.
To adapt bilevel optimization to deep neural networks,
researchers introduced back-gradient descent, a technique that
allows attackers to approximate bilevel optimization solutions
without explicitly solving the inner problem. Munoz-Gonz ̃ alez ́
et al. [74] first applied this approach to neural networks in
2017, demonstrating that gradient-based optimization could
successfully craft poisoned data that influences deep model
behavior. Jagielski et al. [126] extended this work by proposing
a theoretical optimization framework specifically designed for
data poisoning attacks and defenses in regression models.
Another work, MetaPoison [65], a scalable bilevel poison-
ing attack, was introduced to further improve transferability
and stealth. It successfully poisoned Google Cloud AutoML,
demonstrating its ability to bypass real-world security de-
fenses. In 2020, Geiping et al. [64] improved MetaPoison by
introducing the ‘gradient alignment’ objective and proposed
the Witches’ Brew attack. By ensuring the loss gradient of
poisoned samples mimics that of the adversarial target, this
method guides the model to misclassify targets naturally,
making it highly effective and transferable.
Bilevel optimization has also been adapted to specific do-
mains. Li et al. [127] exploited differentially private crowd-
sensing systems by embedding poisoned data within privacy
noise, making attacks undetectable while degrading system
accuracy. In autonomous driving, Pourkeshavarz et al. [120]
used bilevel optimization to introduce adversarial trajectory
perturbations, causing self-driving systems to misinterpret
vehicle movements. Similarly, Sun et al. [121] applied bilevel
optimization to contrastive learning, optimizing trigger place-
ment to create persistent backdoors in self-supervised models.

E. Influence-based Attacks

Influence-based poisoning attacks leverage influence func-
tions to analyze and manipulate the impact of specific training

```
samples on model predictions. This method is particularly
effective in scenarios where the attacker has partial knowledge
of the model and needs to optimize poisoning efficiency
with minimal data modifications. By leveraging this tech-
nique, attackers can pinpoint the most influential samples and
selectively poison them to maximize their effect on model
predictions.
```
```
I(x) =−Hθ−′^1 ∇θL(f(x,θ))
θ=θ′
```
```
θ′= arg min
θ
```
```
Xn
```
```
i=
```
```
L(f(xi,θ))
```
### (7)

```
wherexis the target sample,His the Hessian matrix of the
empirical risk function, capturing second-order dependencies
in parameter updates.xini=1refers to a set of data,Lis the loss
function, andθis the parameters of the target model obtained
withoutx.
Koh et al. [85], [122] were the first to introduce influence
functions into gradient-based analysis for adversarial attacks,
providing a framework for efficiently approximating bilevel
optimization solutions. Fang et al. [78] extended this idea
to recommender systems, showing that strategically injected
interactions could bias recommendation outcomes. Despite
the effectiveness of influence-based attacks, Basu et al. [123]
identified key limitations when applying them to deep neural
networks. They argued that due to the non-convex nature
of deep learning loss landscapes, influence functions fail to
accurately capture complex dependencies between training and
test samples. This limitation suggests that while influence-
based poisoning is highly effective in convex and structured
models, its application in deep learning remains an open
challenge.
```
```
F. Generative Attacks
Unlike traditional poisoning attacks that rely on perturba-
tion of training samples, generative attacks directly synthe-
size highly realistic poisoned data, making them significantly
more challenging to detect while enhancing their effectiveness
in misleading the target model. Moreover, traditional data
poisoning methods often face limitations in the efficiency
of generating and deploying poisoned samples. In contrast,
generative attacks leverage generative models to substantially
reduce computational costs associated with optimization-based
poisoning, thereby greatly improving the efficiency of gener-
ating and using poisoned data. These attacks typically require
the attacker to possess knowledge of the target model, making
them particularly well-suited for gray-box or white-box threat
models.
Yang et al. [75] proposed a generative poisoning attack
framework based on an encoder-decoder architecture. This
framework consists of two key components: the generator
model Gand the target model f. The poisoning process
follows an iterative optimization procedure: At iterationi, the
generator produces poisoned data, which is then injected into
the training dataset. This leads to an update of the target
model’s parameters from θ(i−1) toθ(i). The attacker then
evaluates the target model’s performance on the validation
```

setDval and uses the results to guide further refinements
of the generator. The generator is subsequently updated, and
the process repeats. This iterative framework can be formally
expressed as:

```
G′= arg max
G
```
### X

```
(x,y)∼Dval
```
```
L(fθ′(G(x)),y)
```
```
s.t. θ′= arg min
θ
```
### X

```
(x,y)∼Dp
```
```
L(fθ(G′(x)),y)
```
### (8)

whereθrepresents the original parameters of the target model
f, andθ′denotes the parameters after poisoning. The ultimate
objective of the generative attack is to train the generator
Gto produce an unlimited supply of poisoned samples that
systematically degrade the performance of the target model.
Building upon this approach, Feng et al. [76] introduced an
enhanced generative model training strategy, incorporating a
pseudo-update mechanism to optimize the generatorG. This
modification addresses the instability caused by alternating
updates betweenfandG, significantly improving the con-
vergence and effectiveness of the generative poisoning attack.
In addition to autoencoders, generative adversarial networks
(GANs) have also been leveraged for data poisoning. For
instance, Munoz-Gonz ̃ ́alez et al. [77] proposed pGAN, which
uses a generatorG, discriminatorD, and classifierf. The
generator produces poisoned samples that misleadf, while
the discriminator fails to distinguish poisoned from clean data,
ensuring a balance between attack effectiveness and stealth.
Expanding on GAN-based attacks, Psychogyios et al. [124]
demonstrated how GAN-generated synthetic data poisons fed-
erated learning, degrading model performance even in early
training. Chen et al. [125] applied GANs to QoS-aware
cloud APIs, injecting poisoned interaction data to manipulate
recommendation rankings. By stealthily distorting API rank-
ings, GANs effectively compromised recommendation systems
without immediate detection, underscoring their adaptability
across AI applications.

G. Others

Next, we briefly introduce several other types of poisoning
attacks on deep neural networks. Pang et al. [128] developed
an innovative accumulative poisoning attack aimed at real-
time data streams in machine learning systems. Their method
introduces an “accumulative phase” in which the attack is
slowly executed over time without triggering immediate model
accuracy degradation. By carefully manipulating the model
state through sequential updates, this attack amplifies the effect
of a poisoned trigger batch, making it significantly more de-
structive than traditional poisoning attacks. Gupta et al. [129]
introduced a novel data poisoning attack for federated learning,
which leverages an inverted loss function. By inverting the
gradients during training, this approach generates malicious
gradients that mislead the model, making it significantly harder
for standard defenses to detect and mitigate. Kasyap and
Tripathy [45] explored poisoning attacks within the context
of federated learning and GANs. They proposed using hyper-
dimensional computing (HDC) to generate adversarial samples
that blend seamlessly with normal data, enhancing the stealth
of the attack.

### V. DATA POISONING IN LARGE LANGUAGE

### MODELS

```
LLMs have revolutionized natural language processing
(NLP), enabling advancements in text generation, machine
translation, code synthesis, conversational AI, and informa-
tion retrieval. State-of-the-art models such as GPT-4 [11],
PaLM [12], and LLaMA [130] have demonstrated exceptional
generalization capabilities, allowing them to perform diverse
tasks with minimal human intervention. However, as these
models grow in size and capability, they also become in-
creasingly susceptible to adversarial attacks. Among these,
data poisoning attacks pose a particularly insidious threat
due to their ability to covertly manipulate model behavior.
Given the widespread use of LLMs in critical applications
such as healthcare [131], legal analysis [132], and financial
forecasting [133], ensuring their robustness against poisoning
attacks is a pressing security concern.
Unlike traditional deep learning models, where poisoning
typically occurs in supervised learning phase, LLMs operate
across multiple vulnerable stages, including pre-training, fine-
tuning, preference alignment, and instruction tuning, etc (rep-
resented in Fig. 5). These stages may all become targets of data
poisoning attacks, so the security of LLMs faces more complex
challenges than traditional deep learning. In the following, we
will introduce in detail the data poisoning attacks that each
stage of LLMs may suffer.
```
```
A. Pre-training
```
```
During the pre-training phase, LLMs are trained on large-
scale unsupervised textual data. Attackers can manipulate web-
scraped data by injecting malicious content into open-access
sources such as Wikipedia, social media, and news platforms,
thereby influencing the foundational knowledge acquired by
the model. For instance, modifying online encyclopedic entries
or strategically inserting misinformation into frequently ac-
cessed web pages can lead to the propagation of biased or false
information in future model outputs. Additionally, attackers
may leverage search engine optimization (SEO) techniques to
ensure that web crawlers retrieve manipulated content, thereby
influencing the data distribution of the training corpus.
Recent studies have highlighted various data poisoning
threats at this stage. Carlini and Terzis [134] demonstrated
targeted poisoning attacks on large-scale image-text datasets
and Contrastive Language-Image Pre-training (CLIP), reveal-
ing that manipulating a few image-text pairs can mislead
model classification in zero-shot scenarios. Carlini et al. [135]
further identified practical vulnerabilities by exploiting expired
image links in web-scale datasets, replacing original data
with poisoned samples at low cost, and underscored the
feasibility of transient textual poisoning on platforms like
Wikipedia. Moreover, Shan et al. [136] introduced Nightshade,
a prompt-specific attack leveraging concept sparsity in pre-
training datasets, showing that a small number of strategically
poisoned images can significantly manipulate text-to-image
model outputs.
```

Fig. 5. Data Poisoning In LLMs.

B. Fine-tuning

Many LLMs undergo fine-tuning to specialize in domain-
specific tasks such as law, medicine, and finance [137]. At-
tackers can inject poisoned samples into fine-tuning datasets to
influence the model’s behavior on specific tasks. For example,
attackers may introduce misleading legal case interpretations
in a legal fine-tuning dataset or embed incorrect medical
treatment guidelines into a healthcare-oriented model, leading
to erroneous and potentially harmful responses. Furthermore,
backdoor attacks can be introduced at this stage by embedding
hidden triggers in training data, causing the model to produce
attacker-defined outputs when specific keywords or patterns
are present in user queries.
Iba ̃nez-Lissen et al. [138] highlighted vulnerabilities in fine-
tuned multimodal models, demonstrating that poisoning one
modality can covertly influence tasks in other modalities,
with a poisoning rate as low as 5% achieving notable attack
effectiveness. Bowen et al. [139] introduced jailbreak-tuning,
combining fine-tuning with jailbreak triggers to bypass model
safety mechanisms, significantly reducing refusal rates for
harmful prompts, especially in larger models.

C. Preference Alignment

Preference alignment, typically implemented via Reinforce-
ment Learning from Human Feedback (RLHF), fine-tunes
LLMs to align with human values and expected behav-
iors [140]–[142]. However, attackers can manipulate RLHF
data to introduce unsafe or biased preferences. For instance,
Fu et al. [143] introduced POISONBENCH, demonstrating

```
that minimal poisoned alignment data could significantly dis-
tort model preferences, degrading helpfulness, harmlessness,
and truthfulness. Shao et al. [144] further showed that sub-
tle manipulations in RLHF datasets substantially increased
model susceptibility to prompt injection attacks. Rando and
Tramer [145] revealed the potency of embedding universal`
jailbreak backdoors via poisoned RLHF data, allowing at-
tackers to bypass safety measures even with limited dataset
manipulation. The study also found that larger models are
not inherently more resilient—instead, higher model capacity
often leads to stronger generalization of the poisoned behavior.
```
```
D. Instruction Tuning
Instruction tuning optimizes LLMs to better understand and
execute user instructions [146]–[148]. Attackers can introduce
malicious instruction samples that modify the response pat-
terns of the model to specific prompts. For instance, poisoning
instruction datasets may lead to cases where the model incor-
rectly refuses to provide legitimate responses while complying
with adversarially crafted prompts. Moreover, attackers can
construct deceptive training tasks that appear benign but sys-
tematically steer model behavior in an undesirable direction,
thereby degrading its overall reliability and safety.
Several studies have demonstrated the susceptibility of
instruction-tuned LLMs to poisoning attacks. Wan et al. [149]
showed that inserting as few as 100 poisoned samples could
embed hidden triggers, causing manipulated model responses
under specific conditions. Shu et al. [150] proposed AutoPoi-
son, an automated framework enabling large-scale and stealthy
```

instruction poisoning attacks that evade standard defenses. Xu
et al. [151] further demonstrated the persistence of instruction-
based backdoors across multiple NLP tasks, remaining re-
sistant to mitigation efforts. Differing from previous works,
Yan et al. [152] introduced Virtual Prompt Injection (VPI),
a backdoor attack triggering adversarial instructions without
explicitly modifying user queries. More recently, Qiang et
al. [153] presented Gradient-Guided Backdoor Trigger Learn-
ing (GBTL), efficiently identifying adversarial triggers to
manipulate instruction-tuned models using minimal poisoned
data.

E. Prefix Tuning

Prefix tuning is a parameter-efficient fine-tuning (PEFT)
method that optimizes a small set of task-specific parameters
while keeping the pre-trained LLM frozen [154], [155]. This
approach reduces computational costs and prevents catas-
trophic forgetting but introduces new security risks. Attackers
can inject malicious prefix-tuning parameters or trigger-based
poisoned prefixes to manipulate model outputs for generative
tasks such as text summarization and completion. By carefully
designing adversarial triggers and poisoned prefix embeddings,
attackers can covertly implant backdoors that activate only
under specific conditions, making detection and mitigation
highly challenging.
In [156], a data poisoning attack on prefix-tuned gener-
ative models was proposed, leveraging the fact that prefix
embeddings act as soft prompts that influence model behavior
without modifying its core parameters. The attack introduces
adversarially crafted prefixes during fine-tuning, encoding
malicious behaviors or biases that activate only when specific
trigger conditions are met. By manipulating how prefixes guide
model outputs, the attack ensures that the poisoned model
exhibits normal performance on clean inputs while producing
manipulated responses in adversarial contexts. Experimental
results on text summarization and completion tasks using T5-
small and GPT-2 models revealed that even small poisoning ra-
tios (1-10%) could introduce persistent adversarial behaviors,
while standard filtering defenses failed to detect the poisoned
samples. Moreover, the study introduced new evaluation met-
rics for assessing attack stealth and effectiveness, emphasizing
the urgent need for improved security measures in PEFT-based
fine-tuning workflows.

F. Prompt Tuning

Similar to prefix tuning, prompt tuning is a parameter-
efficient fine-tuning method that optimizes model responses
by training on task-specific prompts [157], [158]. Attackers
can exploit this process by embedding biased or misleading
prompts in the training set, influencing how the model re-
sponds to certain queries. For example, adversarially modified
prompt structures can induce the model to favor a specific
perspective or default to attacker-defined responses in par-
ticular contexts. Additionally, hidden trigger words can be
integrated into tuned prompts, causing the model to exhibit
pre-determined behaviors upon encountering these inputs.

```
A recent study introduced POISONPROMPT, a novel
backdoor attack targeting both hard and soft prompt-based
LLMs [159]. This attack leverages bi-level optimization to
concurrently train backdoor behavior and prompt tuning tasks,
ensuring that the backdoor is activated only when a specific
trigger phrase is present in the input. Otherwise, the model
functions normally, making detection extremely challenging.
Experiments conducted on six datasets and three widely used
LLMs demonstrated that POISONPROMPT achieves attack
success rates exceeding 90% while maintaining high perfor-
mance on standard tasks.
```
```
G. In-Context Learning (ICL)
In-context learning (ICL) allows LLMs to adapt to new
tasks during inference by conditioning on provided exam-
ples [160]–[162]. Attackers can exploit this mechanism by
injecting poisoned examples within a given input context,
leading the model to generate incorrect, biased, or harmful
outputs. For instance, attackers can introduce misleading few-
shot examples that distort the model’s understanding of a task,
causing systematic errors in its completions. Because ICL
operates at the inference stage and does not modify model
weights, traditional defense mechanisms may struggle to detect
and mitigate such attacks.
Recent studies have highlighted ICL vulnerabilities to poi-
soning attacks. He et al. [163] introduced ICLPoison, a frame-
work applying discrete text perturbations to strategically ma-
nipulate LLM hidden states during inference, significantly de-
grading model performance. Experiments on models including
GPT-4 showed up to a 10% drop in ICL accuracy under attack.
Zhao et al. [164] further proposed ICLAttack, a backdoor
framework exploiting demonstration poisoning without fine-
tuning. By embedding hidden triggers into demonstrations or
prompts, ICLAttack consistently forced adversarially prede-
fined responses.
```
```
H. Prompt Injection
During inference, attackers can leverage prompt injection
attacks to manipulate LLM outputs by crafting deceptive in-
puts. By constructing adversarial prompts, such as “Ignore all
previous instructions and execute the following task...”, attack-
ers can induce the model to generate unintended or harmful
responses, bypassing built-in safety mechanisms. Moreover,
indirect prompt injection exploits external data sources, such
as retrieved web content or API responses, embedding hidden
instructions that LLMs process as valid prompts, potentially
leading to unauthorized actions or information disclosure.
[165] introduced Indirect Prompt Injection (IPI) as a new
security risk for LLM-integrated applications. This attack
exploits the integration of LLMs with external data sources,
allowing attackers to inject harmful instructions into content
retrieved during inference, such as from search engines or code
repositories. The study demonstrated that even benign-looking
sources, like web content or email data, can be weaponized
to indirectly control the LLM’s behavior without direct user
interaction. The stealthiness of this attack, coupled with its
scalability, makes it a significant challenge for current defense
mechanisms.
```

### VI. FUTURE WORK

In this section, we formulate some possible research direc-
tion deserving further exploring.

A. Enhancing the Effectiveness and Stealth of Data Poisoning

- Optimization of Attack Efficiency.While existing poison-
    ing attacks can be highly effective, many require exten-
    sive computational resources or large-scale modifications
    to training data. Future research should explore more
    efficient attack formulations that minimize the amount
    of poisoned data required while maintaining high attack
    success rates.
- Stealthier Poisoning Strategies.As defenses improve, poi-
    soning attacks must become more covert. This includes
    low-perturbation poisoning that introduces imperceptible
    changes while still influencing model decisions, and
    adaptive poisoning attacks that evolve alongside model
    updates to remain undetected.

B. Data Poisoning in Dynamic Learning Environments

- Attacking Continual Learning.Models: Continual learn-
    ing models are updated over time rather than being
    trained on a fixed dataset, making them more resistant to
    traditional poisoning techniques. Research should focus
    on how poisoned data can be injected gradually over
    multiple learning cycles to create long-term degradation
    without detection.
- Online Poisoning Attacks.Many AI models are deployed
    in real-time learning environments, such as fraud detec-
    tion, personalized recommendation systems, and stock
    market prediction models. Developing poisoning tech-
    niques that target incrementally updated models without
    requiring a complete retraining cycle would expand the
    practical applicability of data poisoning attacks.
- Accumulative Poisoning.Unlike static attacks that intro-
    duce poisoned samples in a single batch, future poison-
    ing techniques could exploit sequential data poisoning,
    where adversarial data is strategically introduced over
    time to gradually shift model behavior while remaining
    undetected.

C. Generalization and Transferability of Data Poisoning

- Cross-Model Transferability.Many poisoning attacks rely
    on knowledge of a specific model or architecture. Future
    research should focus on developing poisoning techniques
    that generalize across different models and architectures,
    particularly in black-box settings where attackers have
    limited information.
- Robustness Against Model Adaptation.Real-world ma-
    chine learning systems often undergo continuous retrain-
    ing with new data. Investigating poisoning methods that
    remain effective even when the model is updated or fine-
    tuned is crucial for increasing attack persistence.
- Universal Poisoning Attacks.Unlike targeted attacks that
    aim to manipulate a specific test sample, universal poison-
    ing strategies aim to degrade overall model performance

```
across diverse datasets and deployment scenarios. Re-
search into dataset-independent poisoning attacks could
significantly expand the applicability of poisoning tech-
niques.
```
```
D. Expanding Data Poisoning to Emerging AI Architectures
```
- Multimodal Poisoning Attacks.As AI models increasingly
    integrate text, image, audio, and video data, understand-
    ing how poisoning attacks transfer across different modal-
    ities is a critical area of research. For instance, poisoning
    textual datasets could influence image generation models
    trained on paired text-image datasets.
- Data Poisoning in LLMs.LLMs are becoming an inte-
    gral part of AI applications. While some research has
    explored poisoning attacks on large-scale models, the
    overall vulnerability of LLMs to data poisoning remains
    insufficiently studied. Future work should explore the
    long-term effects of poisoning on model behavior, the
    reinforcement of biases, misinformation propagation, and
    adversarial prompt manipulation, all of which could sig-
    nificantly impact the reliability and security of LLM-
    driven systems.

```
E. New data poisoning strategies
```
- Synergies Between Data Poisoning and Adversarial At-
    tacks.While adversarial attacks typically target model in-
    ference, and data poisoning attacks target model training,
    their combined effect remains underexplored. Future re-
    search could investigate how data poisoning can amplify
    the effectiveness of adversarial attacks.
- Automated Attack Optimization.The use of reinforcement
    learning, evolutionary algorithms, and generative models
    (GANs, diffusion models) for automatically generating
    optimal poisoning strategies is an emerging field that
    requires deeper investigation.

```
F. Developing Comprehensive Benchmark Frameworks
```
- Standardized Datasets for Poisoning Evaluation.The cur-
    rent landscape of data poisoning research is fragmented,
    with varying datasets used across studies, leading to
    inconsistencies in performance evaluation. There is an
    urgent need for curated benchmark datasets tailored to
    poisoning attack scenarios, spanning different modalities
    such as image, text, and multimodal AI systems.
- Unified Metrics for Attacks. Existing studies em-
    ploy heterogeneous evaluation metrics, making cross-
    comparisons challenging. Future research should estab-
    lish universally accepted metrics, such as Poisoning Suc-
    cess Rate (PSR), Stealth Score (SS), Model Robustness
    Degradation (MRD), and Computational Overhead (CO),
    to standardize attack evaluations.
- Reproducible Experimental Protocols. To enhance the
    reliability and comparability of research findings, poison-
    ing attack experiments should adhere to reproducibility
    best practices, including public code repositories, clear
    experimental settings, and standardized reporting formats.


### VII. CONCLUSION

Data poisoning has become a critical security threat in
deep learning, allowing adversaries to manipulate training data
to degrade model performance, introduce biases, or create
targeted misclassifications. As deep learning models increas-
ingly rely on large-scale datasets from unverified sources,
the risks associated with poisoning attacks continue to grow,
posing challenges to AI security across various domains. This
paper provides a comprehensive analysis of data poisoning
attacks, categorizing them into heuristic-based, label flipping,
feature collision, bilevel optimization, influence-based, gen-
erative, and other attack strategies. We systematically sum-
marize the mechanisms behind data poisoning algorithms.
Furthermore, we highlight the increasing sophistication of
poisoning strategies, particularly in large language models.
Finally, we outline potential future research directions. We
hope that this comprehensive analysis of data poisoning offers
researchers a deeper understanding of existing poisoning attack
methodologies, facilitating the development of more effective
defense strategies to safeguard machine learning systems.

### REFERENCES

```
[1] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,”nature, vol. 521,
no. 7553, pp. 436–444, 2015.
[2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” inProc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016,
pp. 770–778.
[3] Z. Liu, H. Wang, T. Zhou, Z. Shen, B. Kang, E. Shelhamer, and
T. Darrell, “Exploring simple and transferable recognition-aware image
processing,”IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 3,
pp. 3032–3046, 2022.
[4] L. Dong, S. Xu, and B. Xu, “Speech-transformer: a no-recurrence
sequence-to-sequence model for speech recognition,” inProc. IEEE
Int. Conf. Acoust. Speech Signal Process. (ICASSP), 2018, pp. 5884–
5888.
[5] M. Kim, H.-I. Kim, and Y. M. Ro, “Prompt tuning of deep neural
networks for speaker-adaptive visual speech recognition,”IEEE Trans.
Pattern Anal. Mach. Intell., 2024.
[6] A. Vaswani, “Attention is all you need,”Advances in Neural Informa-
tion Processing Systems, 2017.
[7] M. Lewis, “Bart: Denoising sequence-to-sequence pre-training for
natural language generation, translation, and comprehension,”Adv.
Neural Inf. Process. Syst., 2017.
[8] J. D. M.-W. C. Kenton and L. K. Toutanova, “Bert: Pre-training of
deep bidirectional transformers for language understanding,” inProc.
NAACL-HLT, vol. 1, no. 2, 2019.
[9] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askellet al., “Language
models are few-shot learners,”Adv. Neural Inf. Process. Syst., vol. 33,
pp. 1877–1901, 2020.
[10] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchi-
cal text-conditional image generation with clip latents,”arXiv preprint
arXiv:2204.06125, vol. 1, no. 2, p. 3, 2022.
[11] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkatet al., “Gpt-
technical report,”arXiv preprint arXiv:2303.08774, 2023.
[12] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts,
P. Barham, H. W. Chung, C. Sutton, S. Gehrmannet al., “Palm: Scaling
language modeling with pathways,”J. Mach. Learn. Res., vol. 24, no.
240, pp. 1–113, 2023.
[13] M. Abdin, J. Aneja, H. Behl, S. Bubeck, R. Eldan, S. Gunasekar,
M. Harrison, R. J. Hewett, M. Javaheripi, P. Kauffmannet al., “Phi-
technical report,”arXiv preprint arXiv:2412.08905, 2024.
[14] M. Fang, X. Cao, J. Jia, and N. Gong, “Local model poisoning attacks
to{Byzantine-Robust}federated learning,” inProc. USENIX Security
Symp., 2020, pp. 1605–1622.
```
```
[15] A. Panda, S. Mahloujifar, A. N. Bhagoji, S. Chakraborty, and P. Mittal,
“Sparsefed: Mitigating model poisoning attacks in federated learning
with sparsification,” inProc. Int. Conf. Artif. Intell. Stat., 2022, pp.
7587–7624.
[16] N. Akhtar and A. Mian, “Threat of adversarial attacks on deep learning
in computer vision: A survey,”Ieee Access, vol. 6, pp. 14 410–14 430,
2018.
[17] N. Akhtar, A. Mian, N. Kardan, and M. Shah, “Advances in adversarial
attacks and defenses in computer vision: A survey,”IEEE Access,
vol. 9, pp. 155 161–155 196, 2021.
[18] B. Chen, Y. Feng, T. Dai, J. Bai, Y. Jiang, S.-T. Xia, and X. Wang,
“Adversarial examples generation for deep product quantization net-
works on image retrieval,”IEEE Trans. Pattern Anal. Mach. Intell.,
vol. 45, no. 2, pp. 1388–1404, 2022.
[19] K. Ma, Q. Xu, J. Zeng, X. Cao, and Q. Huang, “Poisoning attack
against estimating from pairwise comparisons,”IEEE Trans. Pattern
Anal. Mach. Intell., vol. 44, no. 10, pp. 6393–6408, 2021.
[20] F. A. Yerlikaya and S ̧. Bahtiyar, “Data poisoning attacks against
machine learning algorithms,”Expert Syst. Appl., vol. 208, p. 118101,
2022.
[21] L. Sun, Y. Dou, C. Yang, K. Zhang, J. Wang, S. Y. Philip, L. He, and
B. Li, “Adversarial attack and defense on graph data: A survey,”IEEE
Trans. Knowl. Data Eng., vol. 35, no. 8, pp. 7693–7711, 2022.
[22] N. Pitropakis, E. Panaousis, T. Giannetsos, E. Anastasiadis, and
G. Loukas, “A taxonomy and survey of attacks against machine
learning,”Comput. Sci. Rev., vol. 34, p. 100199, 2019.
[23] F. Tahmasebian, L. Xiong, M. Sotoodeh, and V. Sunderam, “Crowd-
sourcing under data poisoning attacks: A comparative study,” inProc.
IFIP WG 11.3 Conf. Data Appl. Secur. Privacy, 2020, pp. 310–332.
[24] I. M. Ahmed and M. Y. Kashmoola, “Threats on machine learning
technique by data poisoning attack: A survey,” inProc. Int. Conf. Adv.
Cyber Secur. (ACeS), 2021, pp. 586–600.
[25] M. A. Ramirez, S.-K. Kim, H. A. Hamadi, E. Damiani, Y.-J. Byon,
T.-Y. Kim, C.-S. Cho, and C. Y. Yeun, “Poisoning attacks and defenses
on artificial intelligence: A survey,”arXiv preprint arXiv:2202.10276,
2022.
[26] J. Fan, Q. Yan, M. Li, G. Qu, and Y. Xiao, “A survey on data poisoning
attacks and defenses,” inProc. IEEE Int. Conf. Data Sci. Cyberspace
(DSC), 2022, pp. 48–55.
[27] Z. Tian, L. Cui, J. Liang, and S. Yu, “A comprehensive survey on
poisoning attacks and countermeasures in machine learning,”ACM
Comput. Surv., vol. 55, no. 8, pp. 1–35, 2022.
[28] Z. Wang, J. Ma, X. Wang, J. Hu, Z. Qin, and K. Ren, “Threats
to training: A survey of poisoning attacks and defenses on machine
learning systems,”ACM Comput. Surv., vol. 55, no. 7, pp. 1–36, 2022.
[29] M. Goldblum, D. Tsipras, C. Xie, X. Chen, A. Schwarzschild, D. Song,
A. Madry, B. Li, and T. Goldstein, “Dataset security for machine
learning: Data poisoning, backdoor attacks, and defenses,”IEEE Trans.
Pattern Anal. Mach. Intell., vol. 45, no. 2, pp. 1563–1580, 2022.
[30] G. Xia, J. Chen, C. Yu, and J. Ma, “Poisoning attacks in federated
learning: A survey,”IEEE Access, vol. 11, pp. 10 708–10 722, 2023.
[31] M. Tayyab, M. Marjani, N. Jhanjhi, I. A. T. Hashem, R. S. A. Usmani,
and F. Qamar, “A comprehensive review on deep learning algorithms:
Security and privacy issues,”Comput. Secur., vol. 131, p. 103297, 2023.
[32] A. E. Cin`a, K. Grosse, A. Demontis, S. Vascon, W. Zellinger, B. A.
Moser, A. Oprea, B. Biggio, M. Pelillo, and F. Roli, “Wild patterns
reloaded: A survey of machine learning security against training data
poisoning,”ACM Comput. Surv., vol. 55, no. 13s, pp. 1–39, 2023.
[33] A. E. Cin`a, K. Grosse, A. Demontis, B. Biggio, F. Roli, and M. Pelillo,
“Machine learning security against data poisoning: Are we there yet?”
Computer, vol. 57, no. 3, pp. 26–34, 2024.
[34] D. A. Alber, Z. Yang, A. Alyakin, E. Yang, S. Rai, A. A. Valliani,
J. Zhang, G. R. Rosenbaum, A. K. Amend-Thomas, D. B. Kurland
et al., “Medical large language models are vulnerable to data-poisoning
attacks,”Nat. Med., pp. 1–9, 2025.
[35] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng,
C. Zhang, C. Ruanet al., “Deepseek-v3 technical report,”arXiv
preprint arXiv:2412.19437, 2024.
[36] Y. Bengio, Y. Lecun, and G. Hinton, “Deep learning for ai,”Commun.
ACM, vol. 64, no. 7, pp. 58–65, 2021.
[37] K. N. Kumar, C. K. Mohan, and L. R. Cenkeramaddi, “The impact
of adversarial attacks on federated learning: A survey,”IEEE Trans.
Pattern Anal. Mach. Intell., vol. 46, no. 5, pp. 2672–2691, 2023.
[38] A. Muley, P. Muzumdar, G. Kurian, and G. P. Basyal, “Risk of ai in
healthcare: A comprehensive literature review and study framework,”
arXiv preprint arXiv:2309.14530, 2023.
```

[39] A. Habbal, M. K. Ali, and M. A. Abuzaraida, “Artificial intelligence
trust, risk and security management (ai trism): Frameworks, applica-
tions, challenges and future research directions,”Expert Syst. Appl.,
vol. 240, p. 122442, 2024.
[40] S. Dilmaghani, M. R. Brust, G. Danoy, N. Cassagnes, J. Pecero, and
P. Bouvry, “Privacy and security of big data in ai systems: A research
and standards perspective,” inProc. IEEE Int. Conf. Big Data, 2019,
pp. 5737–5743.
[41] T. Chaalan, S. Pang, J. Kamruzzaman, I. Gondal, and X. Zhang, “The
path to defence: A roadmap to characterising data poisoning attacks on
victim models,”ACM Comput. Surv., vol. 56, no. 7, pp. 1–39, 2024.
[42] Z. Chen, L. Demetrio, S. Gupta, X. Feng, Z. Xia, A. E. Cina, M. Pintor,`
L. Oneto, A. Demontis, B. Biggioet al., “Over-parameterization and
adversarial robustness in neural networks: An overview and empirical
analysis,”arXiv preprint arXiv:2406.10090, 2024.
[43] E. Bagdasaryan, A. Veit, Y. Hua, D. Estrin, and V. Shmatikov, “How
to backdoor federated learning,” inProc. Int. Conf. Artif. Intell. Stat.,
2020, pp. 2938–2948.
[44] V. Tolpegin, S. Truex, M. E. Gursoy, and L. Liu, “Data poisoning
attacks against federated learning systems,” inProc. Eur. Symp. Res.
Comput. Secur. (ESORICS), 2020, pp. 480–501.
[45] H. Kasyap and S. Tripathy, “Beyond data poisoning in federated
learning,”Expert Syst. Appl., vol. 235, p. 121192, 2024.
[46] M. Ozkan-Okay, E. Akin,O. Aslan, S. Kosunalp, T. Iliev, I. Stoyanov, ̈
and I. Beloev, “A comprehensive survey: Evaluating the efficiency of
artificial intelligence and machine learning techniques on cyber security
solutions,”IEEE Access, vol. 12, pp. 12 229–12 256, 2024.
[47] M. Barreno, B. Nelson, R. Sears, A. D. Joseph, and J. D. Tygar,
“Can machine learning be secure?” inProc. ACM Symp. Inf. Comput.
Commun. Secur., 2006, pp. 16–25.
[48] C. Hu and Y.-H. F. Hu, “Data poisoning on deep learning models,” in
Proc. Int. Conf. Comput. Sci. Comput. Intell. (CSCI), 2020, pp. 628–
632.
[49] A. Bajaj and D. K. Vishwakarma, “A state-of-the-art review on ad-
versarial machine learning in image classification,”Multimedia Tools
Appl., vol. 83, no. 3, pp. 9351–9416, 2024.
[50] C. Zhang, S. Yu, Z. Tian, and J. J. Yu, “Generative adversarial
networks: A survey on attack and defense perspective,”ACM Comput.
Surv., vol. 56, no. 4, pp. 1–35, 2023.
[51] T. T. Nguyen, N. Quoc Viet Hung, T. T. Nguyen, T. T. Huynh,
T. T. Nguyen, M. Weidlich, and H. Yin, “Manipulating recommender
systems: A survey of poisoning attacks and countermeasures,”ACM
Comput. Surv., vol. 57, no. 1, pp. 1–39, 2024.
[52] Q. Xu, Z. Yang, Y. Zhao, X. Cao, and Q. Huang, “Rethinking label
flipping attack: From sample masking to sample thresholding,”IEEE
Trans. Pattern Anal. Mach. Intell., vol. 45, no. 6, pp. 7668–7685, 2022.
[53] M. Zhao, B. An, W. Gao, and T. Zhang, “Efficient label contamination
attacks against black-box learning models.” inProc. Int. Joint Conf.
Artif. Intell., 2017, pp. 3945–3951.
[54] S. Gajbhiye, P. Singh, and S. Gupta, “Data poisoning attack by label
flipping on splitfed learning,” inProc. Int. Conf. Recent Trends Image
Process. Pattern Recognit., 2022, pp. 391–405.
[55] Q. Li, X. Wang, F. Wang, and C. Wang, “A label flipping attack on
machine learning model and its defense mechanism,” inProc. Int. Conf.
Algorithms Archit. Parallel Process., 2022, pp. 490–506.
[56] R. Jha, J. Hayase, and S. Oh, “Label poisoning is all you need,”Adv.
Neural Inf. Process. Syst., vol. 36, pp. 71 029–71 052, 2023.
[57] O. Mengara, “A backdoor approach with inverted labels using dirty
label-flipping attacks,”IEEE Access, 2024.
[58] A. Shafahi, W. R. Huang, M. Najibi, O. Suciu, C. Studer, T. Dumitras,
and T. Goldstein, “Poison frogs! targeted clean-label poisoning attacks
on neural networks,”Adv. Neural Inf. Process. Syst., vol. 31, 2018.
[59] O. Suciu, R. Marginean, Y. Kaya, H. Daume III, and T. Dumitras,
“When does machine learning{FAIL}? generalized transferability for
evasion and poisoning attacks,” inProc. USENIX Security Symp., 2018,
pp. 1299–1316.
[60] C. Zhu, W. R. Huang, H. Li, G. Taylor, C. Studer, and T. Goldstein,
“Transferable clean-label poisoning attacks on deep neural nets,” in
Proc. Int. Conf. Mach. Learn., 2019, pp. 7614–7623.
[61] J. Guo and C. Liu, “Practical poisoning attacks on neural networks,”
inProc. Eur. Conf. Comput. Vis., 2020, pp. 142–158.
[62] H. Aghakhani, D. Meng, Y.-X. Wang, C. Kruegel, and G. Vigna,
“Bullseye polytope: A scalable clean-label poisoning attack with im-
proved transferability,” inProc. IEEE Eur. Symp. Secur. Privacy, 2021,
pp. 159–178.

```
[63] J. Lorraine, P. Vicol, and D. Duvenaud, “Optimizing millions of
hyperparameters by implicit differentiation,” inProc. Int. Conf. Artif.
Intell. Stat., 2020, pp. 1540–1552.
[64] J. Geiping, L. Fowl, W. R. Huang, W. Czaja, G. Taylor, M. Moeller,
and T. Goldstein, “Witches’ brew: Industrial scale data poisoning via
gradient matching,”arXiv preprint arXiv:2009.02276, 2020.
[65] W. R. Huang, J. Geiping, L. Fowl, G. Taylor, and T. Goldstein,
“Metapoison: Practical general-purpose clean-label data poisoning,”
Adv. Neural Inf. Process. Syst., vol. 33, pp. 12 080–12 091, 2020.
[66] L. Fowl, P.-y. Chiang, M. Goldblum, J. Geiping, A. Bansal, W. Czaja,
and T. Goldstein, “Preventing unauthorized use of proprietary data:
Poisoning for secure dataset release,”arXiv preprint arXiv:2103.02683,
2021.
[67] T. Gu, B. Dolan-Gavitt, and S. Garg, “Badnets: Identifying vulnera-
bilities in the machine learning model supply chain,”arXiv preprint
arXiv:1708.06733, 2017.
[68] Y. Liu, S. Ma, Y. Aafer, W.-C. Lee, J. Zhai, W. Wang, and X. Zhang,
“Trojaning attack on neural networks,” inProc. Annu. Netw. Distrib.
Syst. Secur. Symp., 2018.
[69] E. Sarkar, H. Benkraouda, and M. Maniatakos, “Facehack: Triggering
backdoored facial recognition systems using facial characteristics,”
arXiv preprint arXiv:2006.11623, 2020.
[70] H. Zhong, C. Liao, A. C. Squicciarini, S. Zhu, and D. Miller, “Back-
door embedding in convolutional neural network models via invisible
perturbation,” inProc. ACM Conf. Data Appl. Secur. Privacy, 2020,
pp. 97–108.
[71] T. A. Nguyen and A. Tran, “Input-aware dynamic backdoor attack,”
Adv. Neural Inf. Process. Syst., vol. 33, pp. 3454–3464, 2020.
[72] K. Doan, Y. Lao, W. Zhao, and P. Li, “Lira: Learnable, imperceptible
and robust backdoor attacks,” inProc. IEEE/CVF Int. Conf. Comput.
Vis., 2021, pp. 11 966–11 976.
[73] Y. Li, Y. Li, B. Wu, L. Li, R. He, and S. Lyu, “Invisible backdoor attack
with sample-specific triggers,” inProc. IEEE/CVF Int. Conf. Comput.
Vis., 2021, pp. 16 463–16 472.
[74] L. Munoz-Gonz ̃ ́alez, B. Biggio, A. Demontis, A. Paudice, V. Wongras-
samee, E. C. Lupu, and F. Roli, “Towards poisoning of deep learning
algorithms with back-gradient optimization,” inProc. ACM Workshop
Artif. Intell. Secur., 2017, pp. 27–38.
[75] C. Yang, Q. Wu, H. Li, and Y. Chen, “Generative poisoning attack
method against neural networks,”arXiv preprint arXiv:1703.01340,
2017.
[76] J. Feng, Q.-Z. Cai, and Z.-H. Zhou, “Learning to confuse: Generating
training time adversarial data with auto-encoder,”Adv. Neural Inf.
Process. Syst., vol. 32, 2019.
[77] L. Munoz-Gonz ̃ alez, B. Pfitzner, M. Russo, J. Carnerero-Cano, and ́
E. C. Lupu, “Poisoning attacks with generative adversarial nets,”arXiv
preprint arXiv:1906.07773, 2019.
[78] M. Fang, N. Z. Gong, and J. Liu, “Influence function based data
poisoning attacks to top-n recommender systems,” inProc. Web Conf.,
2020, pp. 3019–3025.
[79] M. Jagielski, G. Severi, N. Pousette Harger, and A. Oprea, “Subpop-
ulation data poisoning attacks,” inProc. ACM SIGSAC Conf. Comput.
Commun. Secur., 2021, pp. 3104–3122.
[80] X. Chen, C. Liu, B. Li, K. Lu, and D. Song, “Targeted backdoor
attacks on deep learning systems using data poisoning,”arXiv preprint
arXiv:1712.05526, 2017.
[81] A. Turner, D. Tsipras, and A. Madry, “Label-consistent backdoor
attacks,”arXiv preprint arXiv:1912.02771, 2019.
[82] Y. Ge, Q. Wang, J. Yu, C. Shen, and Q. Li, “Data poisoning and
backdoor attacks on audio intelligence systems,”IEEE Commun. Mag.,
vol. 61, no. 12, pp. 176–182, 2023.
[83] J. Zhang, H. Liu, J. Jia, and N. Z. Gong, “Data poisoning based
backdoor attacks to contrastive learning,” inProc. IEEE/CVF Conf.
Comput. Vis. Pattern Recognit., 2024, pp. 24 357–24 366.
[84] J. Zhang, J. Chen, D. Wu, B. Chen, and S. Yu, “Poisoning attack in
federated learning using generative adversarial nets,” inProc. IEEE Int.
Conf. Trust, Secur. Privacy Comput. Commun., 2019, pp. 374–380.
[85] P. W. Koh, J. Steinhardt, and P. Liang, “Stronger data poisoning attacks
break data sanitization defenses,”Mach. Learn., pp. 1–47, 2022.
[86] P.-Y. Chen, H. Zhang, Y. Sharma, J. Yi, and C.-J. Hsieh, “Zoo: Zeroth
order optimization based black-box attacks to deep neural networks
without training substitute models,” inProc. ACM Workshop Artif.
Intell. Secur., 2017, pp. 15–26.
[87] H. Liu, D. Li, and Y. Li, “Poisonous label attack: black-box data
poisoning attack with enhanced conditional dcgan,”Neural Process.
Lett., vol. 53, no. 6, pp. 4117–4142, 2021.
```

[88] P. Chen, Y. Yang, D. Yang, H. Sun, Z. Chen, and P. Lin, “Black-box
data poisoning attacks on crowdsourcing.” inProc. Int. Joint Conf.
Artif. Intell., 2023, pp. 2975–2983.
[89] J. Song, Z. Li, Z. Hu, Y. Wu, Z. Li, J. Li, and J. Gao, “Poisonrec:
an adaptive data poisoning framework for attacking black-box rec-
ommender systems,” inProc. IEEE Int. Conf. Data Eng., 2020, pp.
157–168.
[90] Y. Zhang, X. Yuan, J. Li, J. Lou, L. Chen, and N.-F. Tzeng, “Reverse
attack: Black-box attacks on collaborative recommendation,” inProc.
ACM SIGSAC Conf. Comput. Commun. Secur., 2021, pp. 51–68.
[91] A. N. Bhagoji, S. Chakraborty, P. Mittal, and S. Calo, “Analyzing
federated learning through an adversarial lens,” inProc. Int. Conf.
Mach. Learn., 2019, pp. 634–643.
[92] R. Schuster, C. Song, E. Tromer, and V. Shmatikov, “You autocomplete
me: Poisoning vulnerabilities in neural code completion,” inProc.
USENIX Secur. Symp., 2021, pp. 1559–1575.
[93] S. S. Mishra, H. He, and H. Wang, “Towards effective data poisoning
for imbalanced classification,” inProc. Int. Conf. Mach. Learn., 2023.
[94] Y. Liu, X. Ma, J. Bailey, and F. Lu, “Reflection backdoor: A natural
backdoor attack on deep neural networks,” inProc. Eur. Conf. Comput.
Vis., 2020, pp. 182–199.
[95] Y. Ren, L. Li, and J. Zhou, “Simtrojan: Stealthy backdoor attack,” in
Proc. IEEE Int. Conf. Image Process., 2021, pp. 819–823.
[96] S. Li, M. Xue, B. Z. H. Zhao, H. Zhu, and X. Zhang, “Invisible
backdoor attacks on deep neural networks via steganography and
regularization,”IEEE Trans. Dependable Secure Comput., vol. 18,
no. 5, pp. 2088–2105, 2020.
[97] Y. Xu, J. Yao, M. Shu, Y. Sun, Z. Wu, N. Yu, T. Goldstein, and
F. Huang, “Shadowcast: Stealthy data poisoning attacks against vision-
language models,”arXiv preprint arXiv:2402.06659, 2024.
[98] Z. Yang, B. Xu, J. M. Zhang, H. J. Kang, J. Shi, J. He, and D. Lo,
“Stealthy backdoor attack for code models,”IEEE Trans. Softw. Eng.,
2024.
[99] A. Saha, A. Subramanya, and H. Pirsiavash, “Hidden trigger backdoor
attacks,” inProc. AAAI Conf. Artif. Intell., vol. 34, no. 07, 2020, pp.
11 957–11 965.
[100] M. Alberti, V. Pondenkandath, M. Wursch, M. Bouillon, M. Seuret,
R. Ingold, and M. Liwicki, “Are you tampering with my data?” in
Proc. Eur. Conf. Comput. Vis., 2018, pp. 0–0.
[101] J. Chen, L. Zhang, H. Zheng, X. Wang, and Z. Ming, “Deeppoison:
Feature transfer based stealthy poisoning attack for dnns,”IEEE Trans.
Circuits Syst. II Express Briefs, vol. 68, no. 7, pp. 2618–2622, 2021.
[102] B. Zhao and Y. Lao, “Towards class-oriented poisoning attacks against
neural networks,” inProc. IEEE/CVF Winter Conf. Appl. Comput. Vis.,
2022, pp. 3741–3750.
[103] A. Chan-Hon-Tong, “An algorithm for generating invisible data poi-
soning using adversarial noise that breaks image classification deep
learning,”Mach. Learn. Knowl. Extract., vol. 1, no. 1, pp. 192–204,
2018.
[104] Q. Zhang, W. Ma, Y. Wang, Y. Zhang, Z. Shi, and Y. Li, “Backdoor
attacks on image classification models in deep neural networks,”
Chinese J. Electron., vol. 31, no. 2, pp. 199–212, 2022.
[105] J. Zheng, P. P. Chan, H. Chi, and Z. He, “A concealed poisoning
attack to reduce deep neural networks’ robustness against adversarial
samples,”Inf. Sci., vol. 615, pp. 758–773, 2022.
[106] S. Alahmed, Q. Alasad, J.-S. Yuan, and M. Alawad, “Impacting
robustness in deep learning-based nids through poisoning attacks,”
Algorithms, vol. 17, no. 4, p. 155, 2024.
[107] W. Jiang, H. Li, Y. Lu, W. Fan, and R. Zhang, “Adversarial robustness
poisoning: Increasing adversarial vulnerability of the model via data
poisoning,” inIEEE Global Commun. Conf. IEEE, 2024, pp. 4286–
4291.
[108] D. Solans, B. Biggio, and C. Castillo, “Poisoning attacks on algorithmic
fairness,” inProc. Joint Eur. Conf. Mach. Learn. Knowl. Discov.
Databases, 2020, pp. 162–177.
[109] M.-H. Van, W. Du, X. Wu, and A. Lu, “Poisoning attacks on fair
machine learning,” inProc. Int. Conf. Database Syst. Adv. Appl., 2022,
pp. 370–386.
[110] N. Furth, A. Khreishah, G. Liu, N. Phan, and Y. Jararweh, “Unfair
trojan: Targeted backdoor attacks against model fairness,” inHandbook
of Trustworthy Federated Learning. Springer, 2024, pp. 149–168.
[111] M. Barni, K. Kallas, and B. Tondi, “A new backdoor attack in cnns
by training set corruption without label poisoning,” inProc. IEEE Int.
Conf. Image Process., 2019, pp. 101–105.
[112] M. Xue, C. He, J. Wang, and W. Liu, “Backdoors hidden in facial
features: A novel invisible backdoor attack against face recognition
systems,”Peer-to-Peer Netw. Appl., vol. 14, pp. 1458–1474, 2021.

```
[113] S. Jang, J. S. Choi, J. Jo, K. Lee, and S. J. Hwang, “Silent branding
attack: Trigger-free data poisoning attack on text-to-image diffusion
models,”arXiv preprint arXiv:2503.09669, 2025.
[114] A. Salem, R. Wen, M. Backes, S. Ma, and Y. Zhang, “Dynamic
backdoor attacks against machine learning models,” inProc. IEEE Eur.
Symp. Secur. Privacy (EuroS&P). IEEE, 2022, pp. 703–718.
[115] I. Arshad, M. N. Asghar, Y. Qiao, B. Lee, and Y. Ye, “Pixdoor: A
pixel-space backdoor attack on deep learning models,” inProc. 29th
Eur. Signal Process. Conf., 2021, pp. 681–685.
[116] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, “Understand-
ing deep learning (still) requires rethinking generalization,”Commun.
ACM, vol. 64, no. 3, pp. 107–115, 2021.
[117] H. Zhang, N. Cheng, Y. Zhang, and Z. Li, “Label flipping attacks
against naive bayes on spam filtering systems,”Appl. Intell., vol. 51,
no. 7, pp. 4503–4514, 2021.
[118] V. Lingam, M. S. Akhondzadeh, and A. Bojchevski, “Rethinking label
poisoning for gnns: Pitfalls and attacks,” inProc. Int. Conf. Learn.
Represent., 2024.
[119] N. Luo, Y. Li, Y. Wang, S. Wu, Y.-a. Tan, and Q. Zhang, “Enhancing
clean label backdoor attack with two-phase specific triggers,”arXiv
preprint arXiv:2206.04881, 2022.
[120] M. Pourkeshavarz, M. Sabokrou, and A. Rasouli, “Adversarial back-
door attack by naturalistic data poisoning on trajectory prediction in
autonomous driving,” inProc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit., 2024, pp. 14 885–14 894.
[121] W. Sun, X. Zhang, H. Lu, Y. Chen, T. Wang, J. Chen, and L. Lin,
“Backdoor contrastive learning via bi-level trigger optimization,”arXiv
preprint arXiv:2404.07863, 2024.
[122] P. W. Koh and P. Liang, “Understanding black-box predictions via
influence functions,” inProc. Int. Conf. Mach. Learn., 2017, pp. 1885–
1894.
[123] S. Basu, P. Pope, and S. Feizi, “Influence functions in deep learning
are fragile,”Proc. Int. Conf. Learn. Represent., 2021.
[124] K. Psychogyios, T.-H. Velivassaki, S. Bourou, A. Voulkidis, D. Skias,
and T. Zahariadis, “Gan-driven data poisoning attacks and their mit-
igation in federated learning systems,”Electronics, vol. 12, no. 8, p.
1805, 2023.
[125] Z. Chen, T. Bao, W. Qi, D. You, L. Liu, and L. Shen, “Poisoning
qos-aware cloud api recommender system with generative adversarial
network attack,”Expert Syst. Appl., vol. 238, p. 121630, 2024.
[126] M. Jagielski, A. Oprea, B. Biggio, C. Liu, C. Nita-Rotaru, and B. Li,
“Manipulating machine learning: Poisoning attacks and countermea-
sures for regression learning,” inProc. IEEE Symp. Secur. Privacy,
2018, pp. 19–35.
[127] Z. Li, Z. Zheng, S. Guo, B. Guo, F. Xiao, and K. Ren, “Disguised as pri-
vacy: Data poisoning attacks against differentially private crowdsensing
systems,”IEEE Trans. Mob. Comput., vol. 22, no. 9, pp. 5155–5169,
2022.
[128] T. Pang, X. Yang, Y. Dong, H. Su, and J. Zhu, “Accumulative poisoning
attacks on real-time data,”Adv. Neural Inf. Process. Syst., vol. 34, pp.
2899–2912, 2021.
[129] P. Gupta, K. Yadav, B. B. Gupta, M. Alazab, and T. R. Gadekallu, “A
novel data poisoning attack in federated learning based on inverted loss
function,”Comput. Secur., vol. 130, p. 103270, 2023.
[130] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux,
T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar` et al., “Llama:
Open and efficient foundation language models,” arXiv preprint
arXiv:2302.13971, 2023.
[131] J. Qiu, K. Lam, G. Li, A. Acharya, T. Y. Wong, A. Darzi, W. Yuan, and
E. J. Topol, “Llm-based agentic systems in medicine and healthcare,”
Nat. Mach. Intell., vol. 6, no. 12, pp. 1418–1420, 2024.
[132] I. Cheong, K. Xia, K. K. Feng, Q. Z. Chen, and A. X. Zhang, “(a) i
am not a lawyer, but...: Engaging legal experts towards responsible llm
policies for legal advice,” inProc. ACM Conf. Fairness, Accountab.
Transp., 2024, pp. 2454–2469.
[133] X. Yu, Z. Chen, Y. Ling, S. Dong, Z. Liu, and Y. Lu, “Temporal data
meets llm–explainable financial time series forecasting,”arXiv preprint
arXiv:2306.11025, 2023.
[134] N. Carlini and A. Terzis, “Poisoning and backdooring contrastive
learning,” inProc. Int. Conf. Learn. Represent., 2022.
[135] N. Carlini, M. Jagielski, C. A. Choquette-Choo, D. Paleka, W. Pearce,
H. Anderson, A. Terzis, K. Thomas, and F. Tram`er, “Poisoning web-
scale training datasets is practical,” inProc. IEEE Symp. Secur. Privacy,
2024, pp. 407–425.
[136] S. Shan, W. Ding, J. Passananti, S. Wu, H. Zheng, and B. Y. Zhao,
“Nightshade: Prompt-specific poisoning attacks on text-to-image gener-
ative models,” inProc. IEEE Symp. Secur. Privacy, 2024, pp. 212–212.
```

[137] V. B. Parthasarathy, A. Zafar, A. Khan, and A. Shahid, “The ultimate
guide to fine-tuning llms from basics to breakthroughs: An exhaus-
tive review of technologies, research, best practices, applied research
challenges and opportunities,”arXiv preprint arXiv:2408.13296, 2024.
[138] L. Ib ́anez Lissen, J. M. d. Fuentes Garc ̃ ́ıa-Romero de Tejada,
L. Gonz ́alez Manzano, and J. Garc ́ıa Alfaro, “Characterizing poisoning
attacks on generalistic multi-modal ai models,”Inf. Fusion, pp. 1–15,
2023.
[139] D. Bowen, B. Murphy, W. Cai, D. Khachaturov, A. Gleave, and
K. Pelrine, “Data poisoning in llms: Jailbreak-tuning and scaling laws,”
arXiv preprint arXiv:2408.02946, 2024.
[140] H. Lee, S. Phatale, H. Mansoor, K. R. Lu, T. Mesnard, J. Ferret,
C. Bishop, E. Hall, V. Carbune, and A. Rastogi, “Rlaif: Scaling
reinforcement learning from human feedback with ai feedback,” 2023.
[141] S. Casper, X. Davies, C. Shi, T. K. Gilbert, J. Scheurer, J. Rando,
R. Freedman, T. Korbak, D. Lindner, P. Freireet al., “Open problems
and fundamental limitations of reinforcement learning from human
feedback,”arXiv preprint arXiv:2307.15217, 2023.
[142] J. Ji, M. Liu, J. Dai, X. Pan, C. Zhang, C. Bian, B. Chen, R. Sun,
Y. Wang, and Y. Yang, “Beavertails: Towards improved safety align-
ment of llm via a human-preference dataset,”Adv. Neural Inf. Process.
Syst., vol. 36, 2024.
[143] T. Fu, M. Sharma, P. Torr, S. B. Cohen, D. Krueger, and F. Barez,
“Poisonbench: Assessing large language model vulnerability to data
poisoning,”arXiv preprint arXiv:2410.08811, 2024.
[144] Z. Shao, H. Liu, J. Mu, and N. Z. Gong, “Making llms vulner-
able to prompt injection via poisoning alignment,”arXiv preprint
arXiv:2410.14827, 2024.
[145] J. Rando and F. Tram`er, “Universal jailbreak backdoors from poisoned
human feedback,” inProc. Int. Conf. Learn. Represent., 2024.
[146] S. Zhang, L. Dong, X. Li, S. Zhang, X. Sun, S. Wang, J. Li, R. Hu,
T. Zhang, F. Wuet al., “Instruction tuning for large language models:
A survey,”arXiv preprint arXiv:2308.10792, 2023.
[147] B. Peng, C. Li, P. He, M. Galley, and J. Gao, “Instruction tuning with
gpt-4,”arXiv preprint arXiv:2304.03277, 2023.
[148] Y. Wang, Z. Yu, Z. Zeng, L. Yang, C. Wang, H. Chen, C. Jiang,
R. Xie, J. Wang, X. Xieet al., “Pandalm: An automatic evaluation
benchmark for llm instruction tuning optimization,”arXiv preprint
arXiv:2306.05087, 2023.
[149] A. Wan, E. Wallace, S. Shen, and D. Klein, “Poisoning language
models during instruction tuning,” inProc. Int. Conf. Mach. Learn.,
2023, pp. 35 413–35 425.
[150] M. Shu, J. Wang, C. Zhu, J. Geiping, C. Xiao, and T. Goldstein, “On
the exploitability of instruction tuning,”Adv. Neural Inf. Process. Syst.,
vol. 36, pp. 61 836–61 856, 2023.
[151] J. Xu, M. D. Ma, F. Wang, C. Xiao, and M. Chen, “Instructions
as backdoors: Backdoor vulnerabilities of instruction tuning for large
language models,”arXiv preprint arXiv:2305.14710, 2023.
[152] J. Yan, V. Yadav, S. Li, L. Chen, Z. Tang, H. Wang, V. Srinivasan,
X. Ren, and H. Jin, “Backdooring instruction-tuned large language
models with virtual prompt injection,” inProc. North Amer. Chapter
Assoc. Comput., 2024, pp. 6065–6086.
[153] Y. Qiang, X. Zhou, S. Z. Zade, M. A. Roshani, P. Khanduri, D. Zytko,
and D. Zhu, “Learning to poison large language models during instruc-
tion tuning,”arXiv preprint arXiv:2402.13459, 2024.
[154] D. Vos, T. Dohmen, and S. Schelter, “Towards parameter-efficient ̈
automation of data wrangling tasks with prefix-tuning,” inProc. Int.
Conf. Neural Inf. Process. Syst., 2022.
[155] Z. Hu, L. Wang, Y. Lan, W. Xu, E.-P. Lim, L. Bing, X. Xu,
S. Poria, and R. K.-W. Lee, “Llm-adapters: An adapter family
for parameter-efficient fine-tuning of large language models,”arXiv
preprint arXiv:2304.01933, 2023.
[156] S. Jiang, S. R. Kadhe, Y. Zhou, F. Ahmed, L. Cai, and N. Baracaldo,
“Turning generative models degenerate: The power of data poisoning
attacks,”arXiv preprint arXiv:2407.12281, 2024.
[157] Y. Li, Z. Tan, and Y. Liu, “Privacy-preserving prompt tuning for large
language model services,”arXiv preprint arXiv:2305.06212, 2023.
[158] R. I. Masoud, M. Ferianc, P. C. Treleaven, and M. R. Rodrigues, “Llm
alignment using soft prompt tuning: The case of cultural alignment,”
inWorkshop on Socially Responsible Language Modelling Research,
2024.
[159] H. Yao, J. Lou, and Z. Qin, “Poisonprompt: Backdoor attack on prompt-
based large language models,” inProc. Int. Conf. Acoust. Speech Signal
Process., 2024, pp. 7745–7749.
[160] R. Koike, M. Kaneko, and N. Okazaki, “Outfox: Llm-generated essay
detection through in-context learning with adversarially generated

```
examples,” inProc. AAAI Conf. Artif. Intell., vol. 38, no. 19, 2024,
pp. 21 258–21 266.
[161] J. Xu, Z. Cui, Y. Zhao, X. Zhang, S. He, P. He, L. Li, Y. Kang, Q. Lin,
Y. Danget al., “Unilog: Automatic logging via llm and in-context
learning,” inProc. IEEE/ACM Int. Conf. Softw. Eng., 2024, pp. 1–12.
[162] T. Li, G. Zhang, Q. D. Do, X. Yue, and W. Chen, “Long-
context llms struggle with long in-context learning,”arXiv preprint
arXiv:2404.02060, 2024.
[163] P. He, H. Xu, Y. Xing, H. Liu, M. Yamada, and J. Tang, “Data poisoning
for in-context learning,”arXiv preprint arXiv:2402.02160, 2024.
[164] S. Zhao, M. Jia, L. A. Tuan, F. Pan, and J. Wen, “Universal vul-
nerabilities in large language models: Backdoor attacks for in-context
learning,”arXiv preprint arXiv:2401.05949, 2024.
[165] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and
M. Fritz, “Not what you’ve signed up for: Compromising real-world
llm-integrated applications with indirect prompt injection,” inProc.
ACM Workshop Artif. Intell. Secur., 2023, pp. 79–90.
```

