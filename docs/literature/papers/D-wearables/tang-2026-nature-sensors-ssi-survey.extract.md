Source URL: https://www.nature.com/articles/s44460-025-00010-2
Title: Sensing technologies for silent speech interfaces | Nature SensorsClose bannerClose banner

Skip to main content 

Thank you for visiting nature.com. You are using a browser version with limited support for CSS. To obtain the best experience, we recommend you use a more up to date browser (or turn off compatibility mode in Internet Explorer). In the meantime, to ensure continued support, we are displaying the site without styles and JavaScript.

Advertisement

Nature Sensors 

* View all journals
* Saved research
* Search
* Log in

* Content Explore content
* About the journal
* Publish with us
* Sign up for alerts
* RSS feed

1. nature
2. nature sensors
3. review articles
4. article

 Sensing technologies for silent speech interfaces

Download PDF 

* Review Article
* Published: 15 January 2026

# Sensing technologies for silent speech interfaces

* Chenyu Tang ORCID: orcid.org/0000-0002-6368-56391,
* Liang Qi2,
* Shuo Gao ORCID: orcid.org/0000-0003-3096-47002,3,
* Zibo Zhang ORCID: orcid.org/0009-0000-3685-90691,
* Wentian Yi ORCID: orcid.org/0000-0002-4044-30631,
* Muzi Xu ORCID: orcid.org/0000-0001-6381-98631,
* Edoardo Occhipinti4,
* Yu Pan5 &
* …
* Luigi G. Occhipinti ORCID: orcid.org/0000-0002-9067-25341
Show authors 

_Nature Sensors_ **volume 1**, pages 16–26 (2026) Cite this article 

Save article 

View saved research 

* 9153 Accesses
* 1 Citations
* 2 Altmetric
* Metrics details

### Subjects

* Computer science
* Electrical and electronic engineering

## Abstract

Silent speech interfaces decode speech intent without audible sound, enabling communication in settings where voice is inaccessible, or for individuals with speech impairments. Here we examine how sensing technologies shape the capabilities of silent speech interfaces. We compare off-, on- and in-body sensing modalities, identifying how proximity, coupling stability and invasiveness govern signal fidelity, robustness and user comfort. We highlight key trends, including the rise of flexible bioelectronics, multimodal sensor fusion for artefact resilience, and the growing role of edge artificial intelligence in real-time, low-power decoding. We show that on-body systems currently offer the best balance between accuracy and deployability, whereas in-body approaches provide unmatched neural access for individuals with complete loss of articulation. Looking ahead, advances in multimodal sensing, embedded intelligence and closed-loop architectures are poised to expand silent communication across rehabilitation, daily interaction and human–machine interfaces.

 You have full access to this article via your institution.

Download PDF 

### Similar content being viewed by others

### An in-sensor communication electronic textile for imperceptible and ultrarobust silent speech 

Article Open access 04 June 2026 

### A high-performance neuroprosthesis for speech decoding and avatar control 

Article 23 August 2023 

### All-weather, natural silent speech recognition via machine-learning-assisted tattoo-like electronics 

Article Open access 13 August 2021 

## Main

Silent speech interfaces (SSIs) aim to unlock a new dimension of human communication by decoding speech-related intent without relying on vocalized sound1,2. Rather than viewing speech solely as an audible output, these systems conceptualize it as a complex neuromotor process that originates in the brain, propagates through articulatory musculature and can be sensed and reconstructed through a variety of physiological pathways3. This transformative class of technologies redefines how individuals interact with machines and with each other, offering fundamentally new modes of expression in contexts where acoustic speech is inaccessible, impractical or undesired. From enabling silent interaction in noise-sensitive or privacy-critical environments to restoring communication for individuals affected by stroke, neurodegenerative disease or laryngectomy, SSIs address both everyday and medically underserved needs across society2,4.

At the core of any SSI lies the sensing interface, which governs which signal can be accessed, how reliably it can be recorded and under which constraints it can be deployed. Sensing strategies fall into three major categories—off body, on body and in body—defined not solely by physical placement, but also by the nature and stability of coupling between the sensor and the physiological source of speech-related information (Fig. 1a). Off-body interfaces, such as optical and acoustic sensors, prioritize user comfort and deployability through non-contact or loosely coupled approaches, but are limited by indirect or distal coupling to the articulatory system5,6. On-body sensors, including electromyography (EMG), strain and inertial modules, offer a closer and more stable connection to neuromuscular activity through tight skin contact, but introduce wearability considerations7,8,9. In-body neural interfaces, such as electrocorticography (ECoG) or intracortical microelectrodes, provide direct access to the brain’s speech-generating regions and enable decoding even in the complete absence of peripheral muscle control. However, this advantage comes at the cost of invasiveness and a need for custom-designed solutions or intervention procedures due to patient diversity in their anatomical structure, pathological conditions, biological responses and functional goals, alongside ethical constraints10,11.

**Fig. 1: Sensor-based classification of SSIs and their deployment form factors.**

Fig. 1: Sensor-based classification of SSIs and their deployment form factors.

Full size image

**a**, Sensor modalities in SSIs can be categorized, by the proximity of signal acquisition to the human body, as either off-body device interfaces (for example, optical cameras or ultrasound probes), on-body sensor interfaces (for example, surface EMG, strain sensors, EEG and IMUs) or in-body neural interfaces (for example, ECoG, sEEG or MEAs). These categories represent a continuum between user comfort and signal specificity, ranging from general-purpose wearable devices to highly personalized clinical systems. **b**, Representative deployment examples for each category, aligned with the same taxonomy: off body (smartphones, earbuds and smart glasses), on body (facial patch/tattoo, choker/throat patch, mask and EEG hat) and in body (ECoG, sEEG and MEA implants). The asterisks denote technologies with clinical validation.

These sensing modalities present diverse form factors tailored to different user contexts (Fig. 1b). Off-body sensing has been integrated into earbuds, mobile phones and smart glasses, offering non-contact solutions with high comfort but lower signal specificity. On-body approaches, leveraging biomechanical and bioelectrical sensors, are implemented in smart chokers, facial patches, masks and head-mounted wearables, balancing wearability with stable neuromuscular access. In-body strategies are represented by invasive brain–computer interfaces, such as ECoG grids and intracortical electrodes, some of which have already been applied in clinical speech decoding trials.

Overall, these categories delineate trade-offs among comfort, invasiveness and system complexity. Signal fidelity introduces a further dimension: while not dictated by proximity alone, it depends on modality, device properties (for example, impedance and bandwidth), placement stability and downstream processing. Yet, a broad trend remains—closer interfaces to articulatory or neural sources generally yield higher fidelity, manifested as greater information throughput and error resilience under realistic use. These gradients also map onto application domains: off-body sensors support general-purpose interaction (such as silent command input or privacy-preserving communication); on-body sensors enable early-stage restoration of speech in patients with residual motor function; and in-body systems remain the only current method for decoding continuous, near-real-time speech in completely locked-in individuals.

The timeline of the development of SSI technologies mirrors this stratification (Fig. 2). Early studies explored both camera-based lipreading and surface EMG12,13, with intracortical recording of speech-related brain activity also beginning to show feasibility in the late 1990s14. Since 2020, the field has witnessed the proliferation of flexible sensing materials, integration with consumer electronics, and initial clinical deployments of wearable and implanted systems. Across modalities, advances in spatial and temporal resolution, signal robustness and integration with learning algorithms are reshaping what is feasible (Table 1).

**Fig. 2: Timeline of sensor-driven innovations in SSIs.**

Fig. 2: Timeline of sensor-driven innovations in SSIs.

Full size image

Presented is a timeline of SSI innovations, from the earliest to the latest: EMG-based vowel recognition (1985)12, camera-based lipreading (1985)13, EEG word decoding (1997)14, intracortical microelectrodes (1998)81, ECoG speech decoding (2004)52, ultrasound imaging (2004)82, depth cameras (2012)83, ultra-wideband (UWB) radar (2016)84, camera SSIs for mobile devices (2018)85, full-sentence decoding via ECoG (2019)86, flexible EMG sensors (2020)7, wearable strain sensors (2021)87, TENG-based lipreading (2022)33, flexible strain sensors (2022)8, IMU-based decoding (2023)30 and magnetoelastic silent speech sensors (2024)88. Each milestone is categorized as off body (red), on body (yellow) or in body (blue) based on the specific sensing implementation used in the corresponding silent speech study, rather than the full spectrum of technological variants. The asterisks denote technologies with clinical validation.

**Table 1 Comparison of sensing modalities used in SSIs**

Full size table

As silent speech technologies approach a translational inflection point, a sensing-led perspective becomes essential to reframe system capability, usability and societal reach. The field is rapidly expanding with the introduction of novel sensor modalities, advances in flexible bioelectronics, and early-stage clinical studies. However, it still lacks a comprehensive framework that integrates these developments through the lens of sensing. By positioning sensing as both a constraint and a catalyst, this Review redefines the foundations of silent speech systems. Trade-offs in comfort, fidelity and invasiveness are not just technical considerations but strategic levers that shape adoption and impact.

We classify existing approaches into off-, on- and in-body sensing strategies, each presenting distinct trade-offs in signal fidelity, comfort and clinical relevance. We compare sensor modalities across spatial and temporal resolution, invasiveness and integration potential and trace their development from early vision-based systems to recent advances in flexible bioelectronics and neural interfaces. Finally, we outline emerging directions in sensor–artificial intelligence (AI) integration, real-time decoding and closed-loop systems that are poised to transform communication, rehabilitation and human–machine interaction at scale.

## SSIs using off-body sensors

Off-body sensors enable non-intrusive SSIs by capturing articulatory and physiological signals without direct skin contact. These systems typically rely on optical and acoustic modalities integrated into external devices such as cameras, smartphones, glasses or headsets. Their appeal lies in high user comfort, ease of deployment and compatibility with commodity hardware, making them attractive for scalable and low-burden interaction. However, signal quality in off-body sensing is often susceptible to environmental conditions, such as lighting variation or acoustic interference, and may suffer from indirect coupling to the user’s intent.

Recent advances have broadened the landscape of off-body silent speech systems, transitioning from laboratory-grade setups to increasingly wearable and context-aware platforms. For instance, optical systems have evolved from static RGB cameras to mobile and multi-angle vision modules, whereas acoustic approaches have moved beyond medical ultrasound towards integrated earbud and headphone solutions that leverage subtle biomechanical cues. These innovations mark an important step towards accessible, privacy-preserving and device-integrated SSIs, but challenges remain in achieving consistent performance across diverse real-world settings and user profiles.

### Optical sensing

Optical sensing enables silent speech input by visually capturing articulatory movements, including lip and facial dynamics, through off-body modalities. The most straightforward approach involves RGB cameras, which offer high-resolution visual data for articulator tracking5,15,16. However, this method is inherently sensitive to occlusion, head pose variation and lighting conditions and typically requires a fixed setup, limiting portability and real-world applicability.

To improve deployment flexibility, optical sensing has been integrated into mobile platforms. Front-facing smartphone cameras enable real-time interaction without auxiliary hardware (Fig. 3a)17,18,19, whereas depth-sensing modules enhance robustness against environmental variation by capturing three-dimensional motion profiles, achieving 91.3% within-user accuracy and 74.9% cross-user accuracy on a 30-command vocabulary20. Nevertheless, both solutions remain constrained by frontal positioning and the instability of handheld use.

**Fig. 3: Representative sensor modalities used in SSIs.**

Fig. 3: Representative sensor modalities used in SSIs.

Full size image

**a**–**i**, Schematics illustrating key sensing technologies enabling silent speech decoding, categorized by sensing principle: optical sensing (**a**; a smartphone-based front camera detects articulatory motion, such as lip and jaw movements); ultrasonic sensing (**b**; an ultrasound imaging probe beneath the jaw visualizes tongue motion in real time); IMU sensing (**c**; IMUs distributed at the head, lip and chin track multi-point facial kinematics during articulation); triboelectric sensing (**d**; self-powered wearable sensors detect facial motion through contact electrification); EMG (**e**; tattoo-like epidermal electrodes acquire facial myopotentials associated with silent articulation); strain sensing (**f**; textile strain sensors embedded in a smart choker capture throat deformation patterns); EEG (**g**; in-ear conformal bioelectronics measure brain activity associated with speech imagery); ECoG (**h**; implanted cortical arrays decode neural activity from speech-generating regions in patients with brainstem injury); and MEA (**i**; intracortical electrodes implanted in the motor cortex record neural activity associated with fine motor intention, enabling high-resolution decoding of attempted handwriting and speech imagery). PI, polyimmide; PVC, polyvinyl chloride. Panels adapted from: **d**, ref. 33, CC BY 4.0; **e**, ref. 36, CC BY 4.0; **f**, ref. 42, CC BY 4.0; **g**, ref. 47, CC BY 4.0.

To overcome these limitations, alternative camera placements have been explored. Side-21 and chin-mounted22 optical systems provide more stable tracking and allow for more natural user movements during interaction, with reported performance exceeding 90% accuracy on vocabularies of approximately 50 words. Although challenges remain in ensuring consistent performance across diverse usage scenarios, optical sensing remains a user-friendly and hardware-light strategy that is particularly suited for applications prioritizing convenience and accessibility.

### Acoustic sensing

Acoustic sensing has gained marked traction in SSIs due to its off-body implementation, strong resilience to occlusion and poor lighting, and inherent advantages for preserving user privacy. Although conventional speech decoding relies on external microphones to capture audible voice signals, such methods fall outside the scope of SSIs, which aim to decode speech intent in the absence of vocalized sound. One early-studied silent approach utilizes ultrasound imaging, wherein probes placed beneath the jaw capture fine-grained tongue kinematics with high spatial fidelity (Fig. 3b)23. This approach achieves a 3.6-s decoding speed with \~33% word error rate (WER) on vocabularies of several dozen commands, but its dependence on specialized imaging hardware restricts its practicality for daily use.

To circumvent such hardware constraints, several strategies have turned to commodity devices. Smartphone-based methods, for instance, emit inaudible acoustic signals via built-in speakers and analyse their reflections using onboard microphones24,25,26. The reflected echoes encode articulatory movements of the lips and tongue, allowing the systems to reach >90% word-level accuracy and <10% sentence-level WER across vocabularies spanning from simple commands to short conversational sentences. These systems offer a hardware-light solution, yet often face performance degradation in noisy or dynamic environments.

To enhance robustness and integrate seamlessly into everyday settings, researchers have embedded acoustic sensors into wearable devices. Glasses-mounted systems detect perioral skin deformation27, earbuds capture air-pressure variations within the ear canal6,28 and headphones monitor temporomandibular joint motion29. Each configuration taps into distinct biomechanical cues, collectively enabling silent command recognition with minimal user effort and improved tolerance to ambient noise. Reported implementations have repeatedly achieved >90% accuracy on vocabularies of more than 100 words, delivering performance comparable to smartphone-based methods while offering substantially greater portability. Together, these diverse implementations underscore the versatility of acoustic sensing and highlight its growing relevance as a scalable, non-intrusive pathway for silent speech decoding.

### Developmental trends and outlook

Off-body sensing has evolved from static, laboratory-bound cameras and ultrasound probes to wearable and device-integrated platforms such as smartphones, glasses and earbuds. Reported accuracies above 90% demonstrate feasibility, yet performance remains fragile under real-world lighting, acoustic noise and inter-user variability. Future progress hinges on environmental robustness, cross-user adaptation and edge AI integration, and socially acceptable and privacy-preserving form factors are likely to define scalability. Together, these directions frame off-body systems as the most accessible entry point for widespread SSI adoption.

## SSIs using on-body sensors

Although off-body sensors offer high user comfort and ease of integration, their indirect coupling to physiological signals and sensitivity to environmental noise often limit decoding accuracy in unconstrained scenarios. To address these limitations, on-body sensing has emerged as a compelling alternative, providing closer physical proximity to the articulatory system and enabling more direct access to neuromuscular or biomechanical activity. By attaching directly to the body surface through either flexible materials or compact rigid modules, on-body sensors can achieve motion-resolved and often higher-fidelity capture of speech-related signals, even under motion, occlusion or low-light conditions.

This improvement in signal quality comes with trade-offs. Compared with off-body methods, on-body systems require physical contact, which introduces considerations around comfort, attachment stability and long-term usability. Nevertheless, these challenges are often outweighed by the benefits in scenarios that demand precision and robustness. On-body sensors have therefore gained traction not only in silent communication for daily use but also in clinical contexts, where they assist individuals with impaired speech. These systems have been explored in the decoding of residual muscle activity in patients with dysarthria, laryngectomy or neurodegenerative diseases, offering an accessible and responsive interface where conventional acoustic speech is unavailable. The enhanced signal access and application versatility position on-body sensing as a critical component in the development of both consumer and healthcare-grade SSIs.

### Inertial measurement unit sensing

Inertial measurement units (IMUs), long used in gait and gesture recognition, have only recently gained traction for SSIs due to difficulty resolving fine-scale articulatory kinematics amid head motion. Early systems used facial accelerometer arrays to detect speech-induced vibrations, achieving high accuracy (94.65 ± 2.54% in classifying 40 English words) but suffering from head-motion artefacts that constrained real-world use30. Differential sensing paradigms were developed to tackle this challenge, where signals from articulator-mounted IMUs are fused with reference sensors on stable regions such as the forehead or ears to isolate speech-specific motion (Fig. 3c), achieving an average accuracy of 92% across seven users for actual continuous lip-speech recognition on 93 English sentences9. The method offers strong environmental robustness, with immunity to lighting, occlusion and noise, and supports ultra-low-power, consumer-ready form factors. Despite sacrificing spatial specificity compared with bioelectric methods, their motion resilience and wearability position IMUs as scalable solutions for mobile SSI deployment.

### Triboelectric nanogenerator sensing

Triboelectric nanogenerators (TENGs) convert mechanical deformation into electrical signals via contact electrification and electrostatic induction. Their self-powered, low-cost and flexible design makes them attractive for wearable sensing31,32. In SSIs, TENGs enable energy-autonomous detection of articulatory motion. Recent studies have shown that soft, skin-mounted TENG arrays can capture lip dynamics and decode phrases using machine learning (Fig. 3d), yielding an accuracy of 94.5% on 20 words33. Compared with optical methods, which depend on ambient illumination, TENGs provide greater privacy and robustness in variable environments. They can also be seamlessly embedded into daily wear, such as face masks or patches, making them unobtrusive for long-term use. However, their outputs depend on tribo-charge state and contact regime and are sensitive to humidity, sweat and material ageing. Combined with the ultra-high source impedance and load-dependent readout, this leads to non-stationarity and session-to-session gain drift that often requires charge management, high-impedance front ends and periodic calibration.

### EMG sensing

Surface EMG provides non-invasive direct access to muscle activation underlying articulation, enabling robust decoding of speech intent in silent contexts. Compared with inertial or force-based methods, EMG captures signals at the neuromuscular source, offering higher specificity but requiring stable skin contact and being susceptible to motion artefacts34,35. Early systems used rigid electrodes and benchtop acquisition setups, limiting usability. Recent advances in conformal bioelectronics have addressed this. Tattoo-like EMG sensors affixed to facial muscles enabled high-accuracy word-level decoding under dynamic, real-world conditions, recognizing up to 110 daily-use words with an average accuracy of 92.6% (Fig. 3e)36. Additionally, textile-based EMG electrodes integrated into headphone earmuffs captured neuromuscular activity from periauricular and jaw-adjacent muscles and leveraged a multi-channel adaptive decoding network to dynamically weight signal quality. This setup demonstrated high usability and robustness in mobile contexts, achieving 96% accuracy on ten commonly used voice-free control words37. Together, these studies exemplify a convergence of high-fidelity sensing with ergonomic form factors.

### Strain sensing

Strain sensors capture subtle deformations of facial and laryngeal tissue during articulation, providing a direct mechanical interface between user intent and silent speech decoding. Foundational work in wearable strain sensors established the feasibility of stretchable, skin-conformal materials for physiological monitoring across dynamic body surfaces38,39. Translating this concept to SSI, researchers have explored a range of material strategies to balance comfort, signal stability and deployment readiness.

Early demonstrations using ultrathin silicon gauges achieved high sensitivity and fast relaxation times, enabling robust word-level decoding under skin strain (87.53% accuracy among 100 words)8. Building on bioinspired mechanisms, ionic hydrogel sensors mimicked mechanoreceptor transduction to detect throat vibrations without requiring electrical contact, achieving an average accuracy of 95% in the 26-instruction test40. Hybrid systems integrating facial deformation and subcutaneous vibration cues revealed that combined motion signatures carry rich, decodable linguistic information even in noisy settings, demonstrating an average accuracy of 99.05% in classifying basic speech elements (phonemes, tones and words)41.

Textile-based strain sensors further advanced the field by embedding high-resolution sensing into wearable form factors optimized for daily use (Fig. 3f), achieving 95.25% accuracy on 20 frequently used English words42. These systems have evolved from controlled experiments to real-world trials in patients who have suffered strokes, where strain signals were leveraged not only for word decoding but also for capturing emotional context, achieving a 4.2% WER and improving daily communication satisfaction by 55% in five patients recovering from strokes43. Across these developments, strain sensing highlights how material and structural innovations can directly shape the usability and expressiveness of silent speech technologies.

### Electroencephalography sensing

Electroencephalography (EEG) enables non-invasive access to cortical activity via scalp or ear-adjacent electrodes, offering a wearable, surface-based approach to capture neural correlates of speech intent upstream of articulation. However, EEG signals are inherently noisy and spatially diffuse, posing major challenges for speech decoding. Foundational studies demonstrated that event-related potentials44 and modulated sensorimotor rhythms45 can support basic intent detection, laying the groundwork for EEG-based communication.

Recent efforts have redefined the role of EEG in silent speech decoding by innovating at the sensor and system levels. Ear-centred46 and in-ear EEG devices (Fig. 3g)47 have matched traditional scalp setups in decoding accuracy while enhancing comfort and form factor. Meanwhile, to mitigate physiological and motion artefact limitations48, recent studies have explored integrated multimodal platforms combining EEG with additional modalities49,50. By providing complementary spatial and physiological signals (for example, inertial, haemodynamic or muscular activity), these systems enable artefact identification and weighting in decoding pipelines, thereby improving robustness. Moving beyond command classification, large-scale studies now leverage contrastive learning to align EEG signals with deep speech representations51, enabling zero-shot decoding of perceived sentences without retraining on specific vocabularies, with an average accuracy of \~41% (up to 80% in some individuals) over more than 1,000 candidate segments.

Although EEG is constrained by low signal fidelity and high intersubject variability, its distinct advantage lies in the ability to access pre-articulatory neural activity, offering a uniquely scalable non-invasive modality for early-stage intent decoding. Although current accuracies remain modest compared with peripheral sensing methods, ongoing advances in sensor design and learning architectures highlight the potential of EEG to enable predictive and generalized SSI systems.

### Developmental trends and outlook

On-body sensors have evolved from rigid electrodes to soft, skin-conformal bioelectronics and energy-harvesting textiles, aiming to balance signal fidelity with everyday wearability. However, this proximity introduces vulnerability to motion artefacts, skin impedance drift and long-term comfort challenges. Recent progress increasingly hinges on functionally complementary multimodal fusion. For example, EMG signals capture neuromuscular intent but are motion sensitive, whereas strain sensors track tissue deformation and sensor displacement. Fusing the two allows decoding models to contextualize signal quality, improving robustness under movement.

This shift reframes sensor fusion not as redundancy, but as a deliberate strategy to resolve the fidelity–stability trade-off inherent in wearable decoding. Combined with advances in stretchable substrates and low-power design, on-body platforms are positioned as the most deployable SSI solution, balancing accuracy with resilience in real-world conditions.

## SSIs using in-body sensors

In-body sensing offers the most direct access to the neural substrates of speech, capturing activity from cortical regions responsible for planning and articulation. By implanting electrodes either onto the brain surface or within its depths, these systems bypass peripheral musculature and enable speech decoding in individuals who have lost all voluntary motor output. They are uniquely positioned to support communication in cases of locked-in syndrome or advanced neurodegeneration, where no other interface is viable.

Although in-body approaches provide exceptional signal fidelity and decoding precision, they require invasive procedures, patient-specific adaptation and long-term clinical support. As such, their application remains limited to research settings and highly selected clinical scenarios. Nevertheless, recent progress in electrode miniaturization, signal processing and neurosurgical techniques has expanded the scope of implanted speech interfaces. These systems not only advance assistive communication but also serve as platforms for probing the neural basis of language and developing future brain-centred interaction technologies.

### ECoG sensing

ECoG acquires high-fidelity local field potentials via subdural electrode grids placed over cortical speech regions. Compared with non-invasive EEG, ECoG offers superior spatial resolution and bandwidth with reduced signal attenuation, enabling direct access to the neural substrates of articulation52,53. Its semi-invasive nature positions it between scalp-based and intracortical approaches, making it a clinically viable modality for patients with severe motor speech impairments.

Over the past decade, ECoG-based silent speech systems have progressed from decoding isolated phonemes to reconstructing full sentences in real time54,55. Further evolution reflects a broader shift from offline, trial-based studies to continuous, streaming paradigms that prioritize naturalistic communication (Fig. 3h), achieving large-vocabulary decoding at up to 78 words per minute with \~25% WER56 and enabling online fluent synthesis in 80-ms increments57. Recent efforts emphasize low-latency, high-intelligibility speech synthesis directly from neural activity, marking a transition towards closed-loop brain-to-speech systems. In parallel, the development of high-density micro-electrocorticography arrays has enabled finer spatial resolution and improved signal fidelity, reinforcing a device-level trend towards more precise, information-rich neural interfaces58. As decoding architectures mature and deployment barriers narrow, ECoG stands as the most clinically advanced in-body interface for restoring communication in individuals with profound speech loss.

### Stereo-EEG sensing

Stereo-EEG (sEEG) records intracranial neural activity via depth electrodes implanted in cortical and subcortical regions, offering three-dimensional spatial coverage. Compared with ECoG, which requires craniotomy to place grid or strip electrodes directly on the cortical surface, sEEG electrodes are introduced through stereotactically guided burr holes. This minimally invasive surgical approach generally carries lower perioperative morbidity, despite the deeper implantation sites59,60. This accessibility to deeper structures makes sEEG particularly valuable for capturing both cortical and subcortical elements of speech planning and tone modulation.

Recent advances have enhanced sEEG as a sensing modality by improving both stimulation and recording strategies. Intermediate-frequency protocols increase mapping sensitivity while reducing afterdischarges, enabling more stable and spatially specific probing of language circuits61. Complementing these protocol-level refinements, the release of structured sEEG datasets capturing vocalized, mimed and imagined speech provides a richer basis for modelling the sensor-to-signal relationship in diverse linguistic contexts62. Together, these developments position sEEG as a minimally invasive and increasingly optimized sensing platform for speech decoding.

### Microelectrode array sensing

Intracortical microelectrode arrays (MEAs) offer direct access to the spiking activity of neurons, providing unparalleled temporal resolution and spatial specificity for speech decoding63,64. By penetrating cortical tissue, these sensors can capture fine-scale dynamics of speech motor planning that are not accessible through surface-level recordings.

Initial demonstrations showed that MEAs could support phoneme-level decoding in open-loop paradigms, laying the groundwork for intracortical speech interfaces65. More recently, their integration into closed-loop systems has enabled real-time synthesis of intelligible words and phrases in individuals with severe speech loss (Fig. 3i)66. These advances mark a shift from offline analysis to continuous decoding pipelines aimed at restoring functional communication.

Despite their exceptional signal fidelity, MEAs face practical limitations including surgical invasiveness, long-term biocompatibility and signal degradation over time. Nevertheless, they remain the benchmark for understanding the upper bounds of neural resolution in brain-to-speech interfaces.

### Developmental trends and outlook

In-body interfaces represent the frontier of SSI potential, offering direct access to cortical speech networks and enabling decoding capabilities that are fundamentally beyond the reach of peripheral sensing. Although current systems have yet to match the real-time accuracy of advanced on-body solutions, particularly in healthy users, their unique strength lies in restoring communication when peripheral musculature is no longer viable. This transformative promise, however, is inseparable from profound biological and ethical tensions. Biologically, craniotomy, long-term biocompatibility challenges and foreign-body responses threaten signal stability, and the absence of fully implantable wireless systems increases infection risk and hinders long-term deployment. Ethically, these interfaces engage directly with the neural substrates of language, raising unprecedented concerns around mental privacy, data ownership and informed consent for the continuous decoding of inner speech. Far from ancillary, these risks will fundamentally shape the pace, scope and social acceptability of in-body SSI technologies.

Looking ahead, progress in minimally invasive electrodes, fully sealed wireless closed-loop platforms and adaptive learning architectures may help to reconcile the demands of high decoding fidelity with the requirements of long-term biological safety. At the same time, robust governance frameworks addressing privacy, data rights and user agency will be essential to uphold ethical standards and ensure responsible deployment. Taken together, these technological and ethical developments may allow in-body systems to serve not only as clinical tools for individuals with severe impairments, but also as scientific instruments for exploring and interacting with the neural mechanisms underlying human language.

## Conclusions

SSIs are reshaping the landscape of human communication by enabling speech decoding without audible output. This Review examines the field through the lens of sensing technologies, revealing how sensing configurations fundamentally shape system fidelity, comfort and usability. By categorizing SSIs into off-, on- and in-body modalities, we have highlighted the trade-offs that govern their deployment—from ambient, non-contact interaction to implantable neuroprosthetics.

These sensing choices are not merely technical parameters but strategic levers that determine who can benefit, where systems can operate and how effectively silent speech can be translated into actionable outputs. With the convergence of flexible electronics, neuromuscular decoding and intelligent feedback, SSIs are emerging not only as assistive tools for individuals with speech impairments but also as scalable platforms for future human–machine interaction.

Yet, sensing alone does not determine system capability. As SSIs transition towards real-world deployment, the co-evolution of sensing, embedded hardware and learning algorithms is becoming increasingly central. In this context, sensing acts not only as an input modality, but as the foundation of tightly coupled signal processing pipelines that enable responsive, low-latency interaction.

Signals acquired from off-, on- or in-body sensors must be routed through analogue front ends, microcontrollers and wireless modules. Modern SSI systems increasingly incorporate edge AI processors that support real-time, low-power inference directly on wearable or mobile platforms, minimizing latency and safeguarding privacy67. This hardware–algorithm co-design paradigm underpins recent advances, including throat-mounted acoustic and biomechanical sensors that drive robotic control via silent commands68.

Figure 4 provides a conceptual overview of this integrated pipeline—from acquisition to decoding to feedback. The captured signals span optical, acoustic, biomechanical and bioelectrical domains, each offering distinct trade-offs between fidelity and comfort. For instance, wearable ultrasonic sensors and strain gauges have demonstrated strong signal-to-noise ratios and resilience to ambient interference8,24, in some cases outperforming surface EMG in specificity7. These signals are standardized through pre-processing pipelines and increasingly fused across modalities to improve robustness across users and settings.

**Fig. 4: Conceptual integration of sensing, processing and feedback in future SSIs.**

Fig. 4: Conceptual integration of sensing, processing and feedback in future SSIs.

Full size image

SSIs follow a multi-stage pipeline beginning with the acquisition of articulatory or neural signals via off-, on- or in-body sensors. These signals are first conditioned by hardware circuits comprising analogue front ends (AFEs), microcontrollers (MCUs), wireless modules (such as Bluetooth or WiFi) and power and memory units. Although most systems transmit pre-processed data to external devices for inference, a subset integrates edge AI processors to enable local, low-latency decoding. The captured signals span optical, acoustic, biomechanical and bioelectrical domains, each reflecting distinct aspects of speech-related activity. Decoding is achieved through algorithmic frameworks, including deep learning, transfer learning and contrastive learning, enabling context-aware interpretation across users and settings. The outputs are translated into feedback modalities, such as synthesized speech, robotic actions or haptic cues, forming a closed-loop interface for assistive and interactive applications.

The decoding stage is primarily driven by lightweight deep learning models—convolutional and transformer-based architectures that support near-real-time performance on embedded systems42,69. Transfer learning and few-shot personalization enable rapid adaptation to new vocabularies or users11,70, whereas contrastive and cross-modal learning increasingly bridge silent and vocal speech domains using large-scale audio datasets17,51. These algorithmic advances substantially reduce data requirements and enhance generalization.

The final outputs of SSIs range from text and synthesized speech to direct robotic or digital commands43,68. Real-time feedback, such as haptic cues or voice synthesis, is crucial for enabling closed-loop, interactive communication. Although challenges remain in generalization, energy efficiency and vocabulary breadth, the synergistic development of sensing hardware, embedded inference and learning architectures is rapidly moving SSIs towards widespread real-world application.

Looking ahead, the next wave of hardware advances will centre on sensor evolution, miniaturization and integration. Flexible and stretchable materials will continue to transform sensors into conformal, skin-like interfaces capable of robust acquisition under motion and daily wear71,72,73. The emergence of hybrid systems, merging bioelectrical, biomechanical and acoustic domains, will offer richer signal streams and redundancy to counteract artefacts and variability74. Real-time decoding will be increasingly enabled by edge AI processors embedded in wearable circuits, unlocking low-latency inference and feedback without cloud reliance75. These developments are complemented by ultra-low-power design and energy-autonomous sensors such as TENGs, paving the way for continuous, passive sensing ecosystems. As sensing hardware becomes more discreet, adaptive and multimodal, it will not only reshape how silent speech is captured, but enlarge where and by whom it can be used.

Beyond hardware, the software layer of SSIs is entering a new era driven by embodied intelligence and large-scale models. Algorithmically, we anticipate a shift from task-specific neural networks towards multimodal foundation models that integrate audio, EMG, strain and even neural data through shared embeddings and attention mechanisms76. This will enable cross-modal learning, few-shot personalization and generalization across vocabularies and users. Embedded inference will increasingly adopt neuromorphic or event-driven architectures, optimizing latency and energy consumption for real-world use77,78. Furthermore, SSIs are positioned to serve as a key interface for human body digital twins, where physiological signals are continuously mapped onto real-time avatars for speech, emotion and motor intent, enabling closed-loop interaction across healthcare, social robotics and communication prosthetics79. As algorithms evolve to reflect not just learned data but the embodied dynamics of human expression, the boundary between input and interface will blur.

In the broader societal context, the future of SSIs extends far beyond clinical assistive technology. As silent interfaces become embedded in daily life, supporting unobtrusive interaction in public spaces, privacy-preserving commands in shared environments and silent collaboration in noisy or sensitive contexts, they will extend the boundary of human–human and human–machine communication. More importantly, for individuals with profound speech or motor disabilities, SSIs might offer not just a tool but a restoration of agency and identity, helping to alleviate psychological distress and foster more effective rehabilitation80. By grounding these technologies in inclusive design and responsible innovation, we can ensure that silent speech systems contribute not only to technical progress but also to human dignity, empowerment and equitable access to communication.

### Reporting summary

Further information on research design is available in the Nature Portfolio Reporting Summary linked to this article.

## References

1. Denby, B. et al. Silent speech interfaces. _Speech Commun._ **52**, 270–287 (2010). **This comprehensive review formalizes SSIs as a research field**.  
 Google Scholar
2. Gonzalez-Lopez, J. A. et al. Silent speech interfaces for speech restoration: a review. _IEEE Access_ **8**, 177995–178021 (2020).  
 Google Scholar
3. Guenther, F. H. et al. in _Speech Motor Control in Normal and Disordered Speech_ 29–49 (Oxford Univ. Press, 2004).
4. Silva, A. B. et al. The speech neuroprosthesis. _Nat. Rev. Neurosci._ **25**, 473–492 (2024).  
PubMed PubMed Central CAS  Google Scholar
5. Afouras, T. et al. Deep audio-visual speech recognition. _IEEE Trans. Pattern Anal. Mach. Intell._ **44**, 8717–8727 (2022).  
PubMed  Google Scholar
6. Jin, Y. et al. EarCommand: “hearing” your silent speech commands in ear. _Proc. ACM Interact. Mob. Wearable Ubiquitous Technol._ **6**, 1–28 (2022).  
 Google Scholar
7. Liu, H. et al. An epidermal sEMG tattoo-like patch as a new human–machine interface for patients with loss of voice. _Microsyst. Nanoeng._ **6**, 16 (2020). **This study introduces flexible surface EMG sensors to silent-speech-related human** **–machine interfaces**.  
PubMed PubMed Central CAS  Google Scholar
8. Kim, T. et al. Ultrathin crystalline-silicon-based strain gauges with deep learning algorithms for silent speech interfaces. _Nat. Commun._ **13**, 5815 (2022).  
PubMed PubMed Central CAS  Google Scholar
9. Liu, S. et al. A data-efficient and easy-to-use lip language interface based on wearable motion capture and speech movement reconstruction. _Sci. Adv._ **10**, eado9576 (2024).  
PubMed PubMed Central  Google Scholar
10. Card, N. S. et al. An accurate and rapidly calibrating speech neuroprosthesis. _N. Engl. J. Med._ **391**, 609–618 (2024).  
PubMed PubMed Central  Google Scholar
11. Willett, F. R. et al. A high-performance speech neuroprosthesis. _Nature_ **620**, 1031–1036 (2023).  
PubMed PubMed Central CAS  Google Scholar
12. Sugie, N. & Tsunoda, K. A speech prosthesis employing a speech synthesizer: vowel discrimination from perioral muscle activities. _IEEE Trans. Biomed. Eng._ **32**, 485–490 (1985).  
PubMed CAS  Google Scholar
13. Petajan, E. D. Automatic lipreading to enhance speech recognition. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ 40–47 (IEEE, 1985). **This conference paper describes an off-body silent speech decoding system using camera-based automatic lipreading**.
14. Suppes, P., Lu, Z.-L. & Han, B. Brain wave recognition of words. _Proc. Natl Acad. Sci. USA_ **94**, 14965–14969 (1997).  
PubMed PubMed Central CAS  Google Scholar
15. Assael, Y. M., Shillingford, B., Whiteson, S. & de Freitas, N. LipNet: end-to-end sentence-level lipreading. Preprint at <https://doi.org/10.48550/arXiv.1611.01599> (2016).
16. Wand, M., Koutník, J. & Schmidhuber, J. Lipreading with long short-term memory. In _Proc. 2016 IEEE International Conference on Acoustics, Speech and Signal Processing_ 6115–6119 (IEEE, 2016).
17. Su, Z., Fang, S. & Rekimoto, J. LipLearner: customizable silent speech interactions on mobile devices. In _Proc. 2023 CHI Conference on Human Factors in Computing Systems_ 1–21 (ACM, 2023).
18. Pandey, L. & Arif, A. S. LipType: a silent speech recognizer augmented with an independent repair model. In _Proc. 2021 CHI Conference on Human Factors in Computing Systems_ 1–19 (ACM, 2021).
19. Pandey, L. & Arif, A. S. MELDER: the design and evaluation of a real-time silent speech recognizer for mobile devices. In _Proc. CHI Conference on Human Factors in Computing Systems_ 1–23 (ACM, 2024).
20. Wang, X., Su, Z., Rekimoto, J. & Zhang, Y. Watch your mouth: silent speech recognition with depth sensing. In _Proc. CHI Conference on Human Factors in Computing Systems_ 1–15 (ACM, 2024).
21. Chen, T. et al. C-Face: continuously reconstructing facial expressions by deep learning contours of the face with ear-mounted miniature cameras. In _Proc. 33rd Annual ACM Symposium on User Interface Software and Technology_ 112–125 (ACM, 2020).
22. Zhang, R. et al. SpeeChin: a smart necklace for silent speech recognition. _Proc. ACM Interact. Mob. Wearable Ubiquitous Technol._ **5**, 1–23 (2021).  
CAS  Google Scholar
23. Kimura, N., Kono, M. & Rekimoto, J. SottoVoce: an ultrasound imaging-based silent speech interaction using deep neural networks. In _Proc. 2019 CHI Conference on Human Factors in Computing Systems_ 1–11 (ACM, 2019).
24. Tan, J., Nguyen, C.-T. & Wang, X. SilentTalk: lip reading through ultrasonic sensing on mobile phones. In _Proc. IEEE INFOCOM 2017_ _—IEEE Conference on Computer Communications_ 1–9 (IEEE, 2017).
25. Gao, Y. et al. EchoWhisper: exploring an acoustic-based silent speech interface for smartphone. _Proc. ACM Interact. Mob. Wearable Ubiquitous Technol._ **4**, 1–27 (2020).  
 Google Scholar
26. Zhang, Q. et al. SoundLip: enabling word and sentence-level lip interaction for smart devices. _Proc. ACM Interact. Mob. Wearable Ubiquitous Technol._ **5**, 1–28 (2021).  
CAS  Google Scholar
27. Zhang, R. et al. EchoSpeech: continuous silent speech recognition on minimally-obtrusive eyewear powered by acoustic sensing. In _Proc. 2023 CHI Conference on Human Factors in Computing Systems_ 1–18 (ACM, 2023).
28. Dong, X. et al. ReHEarSSE: recognizing hidden-in-the-ear silently spelled expressions. In _Proc. CHI Conference on Human Factors in Computing Systems_ 1–16 (ACM, 2024).
29. Zhang, R. et al. HPSpeech: silent speech interface for commodity headphones. In _Proc. 2023 International Symposium on Wearable Computers_ 60–65 (ACM, 2023).
30. Kwon, J. et al. Novel three-axis accelerometer-based silent speech interface using deep neural network. _Eng. Appl. Artif. Intell._ **120**, 105909 (2023).  
 Google Scholar
31. Zhou, Q. et al. Triboelectric nanogenerator-based sensor systems for chemical or biological detection. _Adv. Mater._ **33**, 2008276 (2021).  
CAS  Google Scholar
32. Wang, Z. L. On Maxwell’s displacement current for energy and sensors: the origin of nanogenerators. _Mater. Today_ **20**, 74–82 (2017).  
 Google Scholar
33. Lu, Y. et al. Decoding lip language using triboelectric sensors with deep learning. _Nat. Commun._ **13**, 1401 (2022). **This paper reports on a triboelectric-sensor-based silent speech decoding system**.  
PubMed PubMed Central CAS  Google Scholar
34. De Luca, C. J. The use of surface electromyography in biomechanics. _J. Appl. Biomech._ **13**, 135–163 (1997).  
 Google Scholar
35. Phinyomark, A. et al. Feature extraction and selection for myoelectric control based on wearable EMG sensors. _Sensors_ **18**, 1615 (2013).  
 Google Scholar
36. Wang, Y. et al. All-weather, natural silent speech recognition via machine-learning-assisted tattoo-like electronics. _npj Flex.Electron._ **5**, 20 (2021).  
 Google Scholar
37. Tang, C. et al. Wireless silent speech interface using multi-channel textile EMG sensors integrated into headphones. _IEEE Trans. Instrum. Meas._ **74**, 1–10 (2025).  
 Google Scholar
38. Lipomi, D. J. et al. Skin-like pressure and strain sensors based on transparent elastic films of carbon nanotubes. _Nat. Nanotechnol._ **6**, 788–792 (2011).  
PubMed CAS  Google Scholar
39. Amjadi, M. et al. Stretchable, skin-mountable, and wearable strain sensors and their potential applications: a review. _Adv. Funct. Mater._ **26**, 1678–1698 (2016).  
CAS  Google Scholar
40. Xu, S. et al. Force-induced ion generation in zwitterionic hydrogels for a sensitive silent-speech sensor. _Nat. Commun._ **14**, 219 (2023).  
PubMed PubMed Central CAS  Google Scholar
41. Yang, Q. et al. Mixed-modality speech recognition and interaction using a wearable artificial throat. _Nat. Mach. Intell._ **5**, 169–180 (2023).  
 Google Scholar
42. Tang, C. et al. Ultrasensitive textile strain sensors redefine wearable silent speech interfaces with high machine learning efficiency. _npj Flex. Electron._ **8**, 27 (2024).  
 Google Scholar
43. Tang, C. et al. Wearable intelligent throat enables natural speech in stroke patients with dysarthria. Preprint at <https://doi.org/10.48550/arXiv.2411.18266> (2024). **This on-body sensing study demonstrates generalized silent speech decoding directly in patients with speech impairment**.
44. Farwell, L. A. & Donchin, E. Talking off the top of your head: toward a mental prosthesis utilizing event-related brain potentials. _Electroencephalogr. Clin. Neurophysiol._ **70**, 510–523 (1988).  
PubMed CAS  Google Scholar
45. Wolpaw, J. R. & McFarland, D. J. Control of a two-dimensional movement signal by a noninvasive brain–computer interface in humans. _Proc. Natl Acad. Sci. USA_ **101**, 17849–17854 (2004).  
PubMed PubMed Central CAS  Google Scholar
46. Kaongoen, N., Choi, J. & Jo, S. Speech-imagery-based brain–computer interface system using ear-EEG. _J. Neural Eng._ **18**, 016023 (2021).  
PubMed  Google Scholar
47. Wang, Z. et al. Conformal in-ear bioelectronics for visual and auditory brain–computer interfaces. _Nat. Commun._ **14**, 4213 (2023).  
PubMed PubMed Central CAS  Google Scholar
48. Occhipinti, E., Davies, H. J., Hammour, G. & Mandic, D. P. Hearables: artefact removal in ear-EEG for continuous 24/7 monitoring. In _Proc._ _2022 International Joint Conference on Neural Networks_ 1–6 (IEEE, 2022).
49. Mandic, D. P. et al. In your ear: a multimodal hearables device for the assessment of the state of body and mind. _IEEE Pulse_ **14**, 17–23 (2023).  
 Google Scholar
50. Cooney, C. et al. A bimodal deep learning architecture for EEG-fNIRS decoding of overt and imagined speech. _IEEE Trans. Biomed. Eng._ **69**, 1983–1994 (2021).  
 Google Scholar
51. Défossez, A. et al. Decoding speech perception from non-invasive brain recordings. _Nat. Mach. Intell._ **5**, 1097–1107 (2023).  
 Google Scholar
52. Leuthardt, E. C. et al. A brain–computer interface using electrocorticographic signals in humans. _J. Neural Eng._ **1**, 63–71 (2004).  
PubMed  Google Scholar
53. Schalk, G. & Leuthardt, E. C. Brain–computer interfaces using electrocorticographic signals. _IEEE Rev. Biomed. Eng._ **4**, 140–154 (2011).  
PubMed  Google Scholar
54. Angrick, M. et al. Speech synthesis from ECoG using densely connected 3D convolutional neural networks. _J. Neural Eng._ **16**, 036019 (2019).  
PubMed PubMed Central  Google Scholar
55. Moses, D. A. et al. Neuroprosthesis for decoding speech in a paralyzed person with anarthria. _N. Engl. J. Med._ **385**, 217–227 (2021).  
PubMed PubMed Central  Google Scholar
56. Metzger, S. L. et al. A high-performance neuroprosthesis for speech decoding and avatar control. _Nature_ **620**, 1037–1046 (2023).  
PubMed PubMed Central CAS  Google Scholar
57. Littlejohn, K. T. et al. A streaming brain-to-voice neuroprosthesis to restore naturalistic communication. _Nat. Neurosci._ **28**, 1–11 (2025).  
 Google Scholar
58. Duraivel, S. et al. High-resolution neural recordings improve the accuracy of speech decoding. _Nat. Commun._ **14**, 6938 (2023).  
PubMed PubMed Central CAS  Google Scholar
59. Munari, C. et al. Stereo-electroencephalography methodology: advantages and limits. _Acta Neurol. Scand._ **89**, 56–67 (1994).  
 Google Scholar
60. Mullin, J. P. et al. Is SEEG safe? A systematic review and meta-analysis of stereo-electroencephalography-related complications. _Epilepsia_ **57**, 386–401 (2016).  
PubMed  Google Scholar
61. Abarrategui, B. et al. New stimulation procedures for language mapping in stereo-EEG. _Epilepsia_ **65**, 1720–1729 (2024).  
PubMed CAS  Google Scholar
62. He, T. et al. VocalMind: a stereotactic EEG dataset for vocalized, mimed, and imagined speech in tonal language. _Sci. Data_ **12**, 657 (2025).  
PubMed PubMed Central  Google Scholar
63. Schwartz, A. B. et al. Brain-controlled interfaces: movement restoration with neural prosthetics. _Neuron_ **52**, 205–220 (2006).  
PubMed CAS  Google Scholar
64. Flint, R. D. et al. Accurate decoding of reaching movements from field potentials in the human motor cortex. _J. Neural Eng._ **9**, 046006 (2012).  
PubMed PubMed Central  Google Scholar
65. Mugler, E. M. et al. Direct classification of all American English phonemes using signals from functional speech motor cortex. _J. Neural Eng._ **11**, 035015 (2014).  
PubMed PubMed Central  Google Scholar
66. Willett, F. R. et al. High-performance brain-to-text communication via handwriting decoding. _Nature_ **593**, 249–254 (2021).  
PubMed PubMed Central CAS  Google Scholar
67. Matsumura, G. et al. Real-time personal healthcare data analysis using edge computing for multimodal wearable sensors. _Device_ **3**, 100597 (2025).  
 Google Scholar
68. Liu, T. et al. Machine learning-assisted wearable sensing systems for speech recognition and interaction. _Nat. Commun._ **16**, 2363 (2025).  
PubMed PubMed Central CAS  Google Scholar
69. Cai, D. et al. SILENCE: protecting privacy in offloaded speech understanding on resource-constrained devices. _Adv. Neural Inform. Proc. Syst._ **37**, 105928–105948 (2024).  
 Google Scholar
70. Deng, Z. et al. Silent speech recognition based on surface electromyography using a few electrode sites under the guidance from high-density electrode arrays. _IEEE Trans. Instrum. Meas._ **72**, 1–11 (2023).  
 Google Scholar
71. Pang, C. et al. A flexible and highly sensitive strain-gauge sensor using reversible interlocking of nanofibres. _Nat. Mater._ **11**, 795–801 (2012).  
PubMed CAS  Google Scholar
72. Tang, C. et al. A deep learning-enabled smart garment for accurate and versatile monitoring of sleep conditions in daily life. _Proc. Natl Acad. Sci. USA_ **122**, e2420498122 (2025).  
PubMed PubMed Central CAS  Google Scholar
73. Xu, M. et al. Simultaneous isotropic omnidirectional hypersensitive strain sensing and deep learning-assisted direction recognition in a biomimetic stretchable device. _Adv. Mater._ **37**, 2420322 (2025).  
PubMed PubMed Central CAS  Google Scholar
74. Gong, S. et al. Hierarchically resistive skins as specific and multimetric on-throat wearable biosensors. _Nat. Neurosci._ **18**, 889–897 (2023).  
CAS  Google Scholar
75. Xue, C. et al. A CMOS-integrated compute-in-memory macro based on resistive random-access memory for AI edge devices. _Nat. Electron._ **4**, 81–90 (2021).  
CAS  Google Scholar
76. Fei, N. et al. Towards artificial general intelligence via a multimodal foundation model. _Nat. Commun._ **13**, 3094 (2022).  
PubMed PubMed Central CAS  Google Scholar
77. Roy, K. et al. Towards spike-based machine intelligence with neuromorphic computing. _Nature_ **575**, 607–617 (2019).  
PubMed CAS  Google Scholar
78. Wang, S. et al. Memristor-based adaptive neuromorphic perception in unstructured environments. _Nat. Commun._ **15**, 4671 (2024).  
PubMed PubMed Central CAS  Google Scholar
79. Tang, C. et al. A roadmap for the development of human body digital twins. _Nat. Rev. Elect. Eng._ **1**, 199–207 (2024).  
 Google Scholar
80. Zinn, S. et al. The effect of poststroke cognitive impairment on rehabilitation process and functional outcome. _Arch. Phys. Med. Rehab._ **85**, 1084–1090 (2004).  
 Google Scholar
81. Kennedy, P. R. & Bakay, R. A. E. Restoration of neural output from a paralyzed patient by a direct brain connection. _Neuroreport_ **9**, 1707–1711 (1998). **This paper reports on a patient-level in-body neural interface demonstrating direct brain-connected communication in a paralysed individual**.  
PubMed CAS  Google Scholar
82. Denby, B. & Stone, M. Speech synthesis from real-time ultrasound images of the tongue. In _Proc. 2004 IEEE International Conference on Acoustics, Speech, and Signal Processing_ 685–688 (IEEE, 2004). **This conference paper demonstrates silent speech synthesis using real-time ultrasound imaging of the tongue.**
83. Galatas, G., Potamianos, G. & Makedon, F. Audio-visual speech recognition incorporating facial depth information captured by the Kinect. In _Proc. 20th European Signal Processing Conference_ 2714–2717 (IEEE, 2012).
84. Shin, Y. H. & Seo, J. Towards contactless silent speech recognition based on detection of active and visible articulators using IR-UWB radar. _Sensors_ **16**, 1812 (2016).  
PubMed PubMed Central  Google Scholar
85. Sun, K. et al. Lip-Interact: improving mobile device interaction with silent speech commands. In _Proc. 31st Annual ACM Symposium on User Interface Software and Technology_ 581–593 (ACM, 2018).
86. Anumanchipalli, G. K., Chartier, J. & Chang, E. F. Speech synthesis from neural decoding of spoken sentences. _Nature_ **568**, 493–498 (2019). **This paper reports a neural decoder that synthesizes full spoken sentences from ECoG recordings**.  
PubMed PubMed Central CAS  Google Scholar
87. Ravenscroft, D. et al. Machine learning methods for automatic silent speech recognition using a wearable graphene strain gauge sensor. _Sensors_ **22**, 299 (2022).  
 Google Scholar
88. Che, Z. et al. Speaking without vocal folds using a machine-learning-assisted wearable sensing-actuation system. _Nat. Commun._ **15**, 1873 (2024).  
PubMed PubMed Central CAS  Google Scholar

Download references

## Acknowledgements

C.T. acknowledges funding from Endoenergy (grant G119004). S.G. acknowledges funding from the National Natural Science Foundation of China (grant 62171014). W.Y. acknowledges funding from Pragmatic Semiconductor (grant G125298). E.O. is partially funded by the Chan Zuckerberg Initiative DAF (grant 2022-316777), an advised fund of the Silicon Valley Community Foundation. L.G.O. acknowledges funding from the British Council (UK–India Education and Research Initiative grant 45371261), UK Research and Innovation (grants EP/W024284/1 and EP/P027628/1), Endoenergy (grant G119004) and Pragmatic Semiconductor (grant G125298).

## Author information

### Authors and Affiliations

1. Department of Engineering, University of Cambridge, Cambridge, UK  
Chenyu Tang, Zibo Zhang, Wentian Yi, Muzi Xu & Luigi G. Occhipinti
2. School of Instrumentation and Optoelectronic Engineering, Beihang University, Beijing, China  
Liang Qi & Shuo Gao
3. Hangzhou International Innovation Institute, Beihang University, Hangzhou, China  
Shuo Gao
4. Department of Mechanical Engineering, University College London, London, UK  
Edoardo Occhipinti
5. Department of Rehabilitation Medicine, Beijing Tsinghua Changgung Hospital, Tsinghua University, Beijing, China  
Yu Pan

Authors
1. Chenyu Tang  
View author publications  
Search author on:PubMed Google Scholar
2. Liang Qi  
View author publications  
Search author on:PubMed Google Scholar
3. Shuo Gao  
View author publications  
Search author on:PubMed Google Scholar
4. Zibo Zhang  
View author publications  
Search author on:PubMed Google Scholar
5. Wentian Yi  
View author publications  
Search author on:PubMed Google Scholar
6. Muzi Xu  
View author publications  
Search author on:PubMed Google Scholar
7. Edoardo Occhipinti  
View author publications  
Search author on:PubMed Google Scholar
8. Yu Pan  
View author publications  
Search author on:PubMed Google Scholar
9. Luigi G. Occhipinti  
View author publications  
Search author on:PubMed Google Scholar

### Contributions

C.T., S.G. and L.G.O. discussed and conceived of the idea for the paper. C.T., L.Q. and S.G. drafted the manuscript. C.T. and Z.Z. visualized the figures. S.G. and L.G.O. supervised the work. C.T., S.G., W.Y., M.X., E.O., Y.P. and L.G.O. contributed to discussing, writing and revising the article.

### Corresponding authors

Correspondence toShuo Gao or Luigi G. Occhipinti.

## Ethics declarations

### Competing interests

The authors declare no competing interests.

## Peer review

### Peer review information

_Nature Sensors_ thanks Jun Chen and the other, anonymous, reviewer(s) for their contribution to the peer review of this work.

## Additional information

**Publisher’s note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

## Supplementary information

### Reporting Summary (download PDF )

## Rights and permissions

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.

Reprints and permissions

## About this article

### Cite this article

Tang, C., Qi, L., Gao, S. _et al._ Sensing technologies for silent speech interfaces._Nat. Sens._ **1**, 16–26 (2026). https://doi.org/10.1038/s44460-025-00010-2

Download citation

* Received: 14 June 2025
* Accepted: 26 November 2025
* Published: 15 January 2026
* Version of record: 15 January 2026
* Issue date: January 2026
* DOI: https://doi.org/10.1038/s44460-025-00010-2

### Share this article

Anyone you share the following link with will be able to read this content:

Get shareable link

Sorry, a shareable link is not currently available for this article.

Copy shareable link to clipboard

 Provided by the Springer Nature SharedIt content-sharing initiative

 You have full access to this article via your institution.

Download PDF 

Advertisement

## Explore content

* Research articles
* Reviews & Analysis
* News & Comment
* Current issue
* Collections
* Sign up for alerts
* RSS feed

## About the journal

* Aims & Scope
* Journal Information
* About the Editors
* Research Cross-Journal Editorial Team
* Reviews Cross-Journal Editorial Team
* Our publishing models
* Editorial Values Statement
* Editorial Policies
* Content Types
* Contact

## Publish with us

* Submission Guidelines
* For Reviewers
* Language editing services
* Open access funding
* Submit manuscript

## Search

Search articles by subject, keyword or author 

Show results from All journals This journal 

Search 

 Advanced search 

### Quick links

* Explore articles by subject
* Find a job
* Guide to authors
* Editorial policies

 Nature Sensors (_Nat. Sens._)

ISSN 3059-4499 (online)

## nature.com footer links

### About Nature Portfolio

* About us
* Press releases
* Press office
* Contact us

### Discover content

* Journals A-Z
* Articles by subject
* protocols.io
* Nature Index

### Publishing policies

* Nature portfolio policies
* Open access

### Author & Researcher services

* Reprints & permissions
* Research data
* Language editing
* Scientific editing
* Nature Masterclasses
* Research Solutions

### Libraries & institutions

* Librarian service & tools
* Librarian portal
* Open research
* Recommend to library

### Advertising & partnerships

* Advertising
* Partnerships & Services
* Media kits
* Branded content

### Professional development

* Nature Awards
* Nature Careers
* Nature  Conferences

### Regional websites

* Nature Africa
* Nature China
* Nature India
* Nature Japan
* Nature Middle East

* Privacy Policy
* Use of cookies
* Your privacy choices/Manage cookies
* Legal notice
* Accessibility statement
* Terms & Conditions
* Your US state privacy rights

Springer Nature 

© 2026 Springer Nature Limited

Close banner Close 

Nature Briefing AI and Robotics 

Sign up for the _Nature Briefing: AI and Robotics_ newsletter — what matters in AI and robotics research, free to your inbox weekly.

Email address 

Sign up 

I agree my information will be processed in accordance with the _Nature_ and Springer Nature Limited Privacy Policy. 

Close banner Close 

Get the most important science stories of the day, free in your inbox. Sign up for Nature Briefing: AI and Robotics 