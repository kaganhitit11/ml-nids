# HEADER

## Abstract

## Introduction

## Literature Review

[Maybe it is only with the introduction, not a seperate section?]


## Methodology


## Experiments and Results

## Discussion

## Conclusion


## References

## Appendix



# DRAFT: later from here is playground for drafting
Machine-learning–based Network Intrusion Detection Systems (NIDS) are widely deployed to detect malicious activity in high-volume networks, yet their robustness to training-time label corruption remains underexplored. In practice, adversaries may exploit data collection and labeling pipelines to introduce targeted label noise that selectively suppresses detection of high-value intrusions or distorts decision boundaries in critical regions of feature space. Despite the operational relevance of such structured label-poisoning threats, existing evaluations are often limited to single datasets or narrow model classes, leaving cross-dataset vulnerability poorly characterized.

This project will address this gap through a systematic, cross-dataset study of targeted label-poisoning attacks and practical defenses for NIDS. We will evaluate four complementary benchmarks, UNSW-NB15, LYCOS- CUPID, and CIDDS-001, using a representative model suite spanning linear and ensemble baselines and neural architectures (Logistic Regression, Random Forests, MLP, 1D-CNN and RNN). Under bounded label-flip budgets, we will implement structured poisoning strategies including class-hiding, feature-targeted, confidence-/loss-aware, disagreement-based and temporal-window and quantify robustness using accuracy, macro/per-class precision–recall–F1, confusion matrices, and degradation curves versus poisoning rate. 

We will further compare training-time defenses based on loss- and disagreement-driven filtering and reweighting, and systematically characterize how these defenses affect poisoned-data robustness and clean-data performance across datasets, model families, and attack strategies.


abstract:

We have systematcily undertaken a cross-dataset study of targeted label-poisoning attacks and practical defenses for machine-learning–based Network Intrusion Detection Systems (NIDS). Our goal was to prove the vulnerability of NIDS to label-poisoning attacks across multiple datasets and model architectures, and to evaluate the effectiveness of various defense mechanisms. 


# Introduction


In today’s world, edge devices are interconnected over sophisticated networks all around the globe. These networks move trillions of packets every day with various intentions—such as web browsing, video and audio streaming, file transfers, remote administration, financial transactions, and sensor data collection—among many others. 

However, because any device connected to the World Wide Web can be addressed, anybody—especially intruders—can send malicious packets or execute harmful activities, for example jamming normal traffic by launching denial-of-service attacks. Such incidents can have serious consequences, ranging from service outages and data breaches to damage that can affect critical infrastructure and even national-level organizations. 

Thus, in order to preserve the safety of both communicating parties, packets must be investigated thoroughly before being accepted as valuable information bits. For this investigation, network intrusion detection systems (NIDS) have been created. NIDS inspect network traffic and attempt to classify behaviors into benign and malicious, providing protection against potential threats. However, the success of these systems is not trivial. Building an effective NIDS requires detailed experiments, wise design choices, and a deep understanding of everyday network activities carried out by security professionals. 

In practice, NIDS performance is often framed in terms of two types of errors. A false negative—where the system fails to detect an ongoing attack—can allow intruders to compromise systems, steal data, or disrupt services with little or no response. A false positive—where the system flags legitimate traffic as malicious—can overwhelm analysts with alerts, leading to alert fatigue and potentially causing real attacks to be missed in the noise. Balancing these errors is a central challenge in NIDS design and deployment. 


With the increased availability of large datasets and advances in machine learning techniques, efforts on NIDS have intensified. Researchers commonly evaluate their approaches on benchmark datasets such as KDD Cup 99 \cite{kdd99}, NSL-KDD \cite{nslkdd}, UNSW-NB15 \cite{unsw}, and CIC-IDS2017 \cite{cic}, which provide labeled examples of normal and malicious traffic. Machine learning models—including logistic regression, random forests, and neural networks—are trained on these labeled datasets to distinguish benign from malicious traffic. However, this reliance on labeled training data introduces a critical vulnerability: if an adversary can corrupt the labels during data collection or annotation, the resulting model may learn to misclassify traffic in ways that benefit the attacker.

This threat, known as *label poisoning* or *label flipping*, is a training-time attack where an adversary manipulates a fraction of the training labels without altering the underlying features. Unlike adversarial examples that perturb inputs at inference time, label poisoning corrupts the model before deployment. The attack can be *untargeted*—randomly flipping labels to degrade overall accuracy—or *targeted*, where specific attack classes are relabeled as benign to create blind spots in the detector. Targeted label poisoning is particularly dangerous because it allows adversaries to selectively suppress detection of high-value intrusions while maintaining acceptable overall accuracy, making the attack difficult to notice.

Despite the operational relevance of such structured label-poisoning threats, existing evaluations are often limited to single datasets or narrow model classes, leaving cross-dataset vulnerability poorly characterized. Prior work has demonstrated label poisoning in image classification and spam filtering, but systematic studies on NIDS across diverse network traffic distributions remain scarce.

In this work, we address this gap through a comprehensive cross-dataset study of targeted label-poisoning attacks and practical defenses for NIDS. We evaluate four complementary benchmark datasets—UNSW-NB15, CIC-IDS2017, CUPID, and CIDDS-001—selected to represent diverse network environments, attack distributions, and class imbalance characteristics. We employ a representative model suite spanning traditional machine learning (Logistic Regression, Random Forest) and neural architectures (MLP, CNN, and RNN) to understand how model complexity affects vulnerability. Under bounded label-flip budgets (5\%, 10\%, 20\%), we implement five structured poisoning strategies: class hiding, feature-targeted poisoning, influence-aware poisoning, disagreement-based poisoning, and temporal window poisoning (applied exclusively to CUPID due to its temporal structure). We further evaluate removal and reweighting defenses to characterize their effectiveness across attack intensities.

Our experiments reveal alarming vulnerabilities. On CIC-IDS2017, a class hiding attack with only 10\% label poisoning reduced attack recall from 98\% to 16\%, and at 20\% poisoning, the model achieved 0\% attack detection—complete failure to identify any malicious traffic. Critically, overall accuracy remained above 80\% in these scenarios, creating an *accuracy illusion* where the model appears functional but is effectively blind to attacks. Defense mechanisms showed limited effectiveness: removal defense provided some protection at low poisoning rates but collapsed at higher rates, while reweighting defenses were largely ineffective across all scenarios. We also observed significant dataset-specific vulnerability patterns, with CIC-IDS2017 being most susceptible and UNSW-NB15 showing greater resilience.

The contributions of this work are:
- **Systematic cross-dataset evaluation**: We provide the first comprehensive study of label poisoning across four diverse NIDS benchmark datasets with consistent experimental methodology.
- **Attack strategy comparison**: We implement and compare five structured poisoning strategies, identifying class hiding as the most effective attack and characterizing the conditions under which each strategy succeeds.
- **Defense analysis**: We evaluate practical training-time defenses (removal and reweighting) and document their failure modes at higher poisoning rates.
- **The accuracy illusion**: We identify and characterize the dangerous phenomenon where poisoned models maintain high overall accuracy while completely failing to detect attacks—a critical finding for practitioners who rely on accuracy as a health metric.




## Related Work

So far, several lines of work have shaped the field. Early real-time systems such as Bro demonstrated the power of deep, protocol-aware network monitoring and scriptable analysis for detecting intruders  . Sommer and Paxson critically examined the use of machine learning for network intrusion detection, highlighting the practical challenges of anomaly detection in real deployments and arguing for more realistic evaluation assumptions  . At the same time, new datasets such as UNSW‑NB15 were introduced to better capture modern traffic and attack patterns, providing a more comprehensive benchmark for evaluating NIDS. More recently, surveys by Khraisat et al. and by Chou and Jiang have synthesized the landscape of intrusion detection techniques and data-driven methods, respectively, clarifying common design choices, datasets, and open challenges  . Together, these works provide a foundation for understanding both the progress and the remaining difficulties in building robust network intrusion detection systems.







# Turkish draft:

Günümüz dünyasında, her cihazın birbirine bağlı olduğu ağlar bütününde paketler ağlar arasınad haereket etmektedir. Bu paketlerin içeriğinin güvenlği veya paketlerin ulaştığı hedeflerin güvenliği bu interconnected yapı yıllar boyunca genişledikçe daha da önemli hale gelmiştir. Bu nedenle, ağ trafiğini izlemek ve kötü niyetli etkinlikleri tespit etmek için Network Intrusion Detection Systems (NIDS) geliştirilmiştir. NIDS, ağ trafiğini analiz ederek anormal davranışları tespit etmeye çalışır ve böylece potansiyel tehditlere karşı koruma sağlar. Ancak bu sistemlerin başarsıı kullanılan modellerin tasarımına ve eğitildiği verilere bağlıdır. Machine-learning tabanlı NIDS modelleri, büyük veri setlerinden öğrenerek anormal davranışları tespit etme yetenekleri nedeniyle yaygın olarak kullanılmaktadır. Ancak, bu modellerin eğitim sürecinde etiket bozulmasına karşı dayanıklılığı yeterince araştırılmamıştır. Gerçek dünyada, kötü niyetli aktörler veri toplama ve etiketleme süreçlerini manipüle ederek hedefli etiket gürültüsü ekleyebilirler. Bu tür yapılandırılmış etiket zehirleme tehditleri, yüksek değerli saldırıların tespitini engellemek veya özellik uzayının kritik bölgelerinde karar sınırlarını bozmak için kullanılabilir. Bu tür tehditlerin operasyonel önemi göz önüne alındığında, mevcut değerlendirmeler genellikle tek bir veri seti veya dar model sınıflarıyla sınırlıdır ve çapraz veri seti savunmasızlığı yeterince karakterize edilmemiştir. Biz de bu çalışmamızda bu boşluğu doldurmayı amaçladık.


Amacımız verisetlerinin, modellerin ve saldırı stratejilerinin çeşitli kombinasyonları arasında etiket zehirleme saldırılarına karşı dayanıklılığı sistematik olarak incelemekti. Karar kılınan poisoning miktarı ile hasar almış bir dataseti her bir makine öğrenme modeli ile eğitip sonrasında test için aurılmış kısmındaki ... metriklerine baktık





# References

@ARTICLE{Paxson1999-le,
  title     = "Bro: a system for detecting network intruders in real-time",
  author    = "Paxson, Vern",
  abstract  = "We describe Bro, a stand-alone system for detecting network
               intruders in real-time by passively monitoring a network link
               over which the intruder's traffic transits. We give an overview
               of the system's design, which emphasizes high-speed (FDDI-rate)
               monitoring, real-time notification, clear separation between
               mechanism and policy, and extensibility. To achieve these ends,
               Bro is divided into an `event engine' that reduces a
               kernel-filtered network traffic stream into a series of
               higher-level events, and a `policy script interpreter' that
               interprets event handlers written in a specialized language used
               to express a site's security policy. Event handlers can update
               state information, synthesize new events, record information to
               disk, and generate real-time notifications via syslog. We also
               discuss a number of attacks that attempt to subvert passive
               monitoring systems and defenses against these, and give
               particulars of how Bro analyzes the six applications integrated
               into it so far: Finger, FTP, Portmapper, Ident, Telnet and
               Rlogin. The system is publicly available in source code form.",
  journal   = "Comput. Netw.",
  publisher = "Elsevier BV",
  volume    =  31,
  number    = "23-24",
  pages     = "2435--2463",
  month     =  dec,
  year      =  1999,
  language  = "en"
}

@article{Sommer2010OutsideTC,
  title={Outside the Closed World: On Using Machine Learning for Network Intrusion Detection},
  author={Robin Sommer and Vern Paxson},
  journal={2010 IEEE Symposium on Security and Privacy},
  year={2010},
  pages={305-316},
  url={https://api.semanticscholar.org/CorpusID:206578669}
}


@article{Chou2021ASO,
  title={A Survey on Data-driven Network Intrusion Detection},
  author={Dylan Chou and Meng Jiang},
  journal={ACM Computing Surveys (CSUR)},
  year={2021},
  volume={54},
  pages={1 - 36},
  url={https://api.semanticscholar.org/CorpusID:242598167}
}