# aiD project: aRTIFICIAL iNTELLIGENCE for the Deaf

Artificial intelligence opens new frontiers for knowledge on the needs of deaf people and for the creation of 
solutions to improve daily life. The project aimed at the development of 
advanced machine learning technologies applicable to commodity mobile devices to target the majority of potential end users.

The overall aim of the aiD project was to bridge together interdisciplinary research areas and leverage
the latest advances in Machine Learning (ML) to develop and provide a comprehensive suite of vital
solutions that will enable deaf people to effectively communicate and interact with hearing people; this
will significantly contribute to the improvement of their daily social life and opportunities of
participation in various social activities. To realize such a vision, aiD is realized the following
technologies:

1. Sign Language Translation: By leveraging recent advances in Transformer-based architectures, we developed a novel 
stochastic variant towards Sign Language Translation. 
2. Automatic Speech Recognition: For automatic speech recognition, we turned to recent advances in unsupervised cross-lingual models.
3.  Text-to-Speech: For producing realistic synthetic speech, we turned to SOTA architectures, namely Tacotron-2.
4. Sign Language Production: We leverage the latest advances in stable diffusion models towards efficient and realistict production 
of Sign Language videos from text. 

All the implementations can be found in their respective folders. For training the models we leveraged benchmark datasets 
for each respective task, while also taking advantage of the datasets produced and curated in the context of the aiD project.
These are publicly available for further research:

1. Greek Elementary Sign Language Dataset: https://zenodo.org/records/5810460
2. Greek News Sign Language Dataset - Part A: https://zenodo.org/records/5810908


## Acknowledgements
This  research  was supported  by  the  the European Union’s Horizon2020 research and innovation program,
under grant agreement  No  872139,  project  aiD.

## Papers
    
    @InProceedings{sergis2021,
        author="Nicolaou, Sergis and Mavrides, Lambros and Tryfou, Georgina and Tolias, Kyriakos and Panousis, Konstantinos and Chatzis, Sotirios
        and Theodoridis, Sergios",
        title="Dialog Speech Sentiment Classification for Imbalanced Datasets",
        booktitle="Speech and Computer",
        year="2021",
        publisher="Springer International Publishing",
        pages="460--471", 
    }

    @InProceedings{Voskou_2021_ICCV,
        author    = {Voskou, Andreas and Panousis, Konstantinos P. and Kosmopoulos, Dimitrios and Metaxas, Dimitris N. and Chatzis, Sotirios},
        title     = {Stochastic Transformer Networks With Linear Competing Units: Application To End-to-End SL Translation},
        booktitle = {Proc. ICCV},
        year      = {2021},
    }

    @InProceedings{panousis_isvc,
        author="Panousis, Konstantinos P. and Chatzis, Soritios and Theodoridis, Sergios",
        title="Variational Conditional Dependence Hidden Markov Models for Skeleton-Based Action Recognition",
        booktitle="Advances in Visual Computing",
        year="2021",
        publisher="Springer International Publishing",
        pages="67--80",
    }

    @InProceedings{pmlr-v130-panousis21a,
        title = { Local Competition and Stochasticity for Adversarial Robustness in Deep Learning },   
        author = {Panousis, Konstantinos and Chatzis, Sotirios and Alexos, Antonios and Theodoridis, Sergios},
        booktitle = {Proceedings of The 24th International Conference on Artificial Intelligence and Statistics},
        pages = {3862--3870},
        year = {2021}
    }


    @article{Panousis_Antoniadis_Chatzis_2022, 
        title={Competing Mutual Information Constraints with Stochastic 
        Competition-Based Activations for Learning Diversified Representations}, 
        author={Panousis, Konstantinos P. and Antoniadis, Anastasios and Chatzis, Sotirios}, 
        booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
        year={2022},  
        pages={7931-7940}
    }

    @InProceedings{pmlr-v162-kalais22a,
      title = 	 {Stochastic Deep Networks with Linear Competing Units for Model-Agnostic Meta-Learning},
      author =       {Kalais, Konstantinos and Chatzis, Sotirios},
      booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
      pages = 	 {10586--10597},
      year = 	 {2022}
    }

    @InProceedings{panousis21bdl,
      title={Stochastic Local Winner-Takes-All Networks Enable Profound Adversarial Robustness},
      author={Panousis, Konstantinos P and Chatzis, Sotirios},
      booktitle={NIPS Bayesian Deep Learning (BDL) Workshop},
      year={2022}
    }


    @InProceedings{Voskou_2023_ICCV,
        author    = {Voskou, Andreas and Panousis, Konstantinos P. and Partaourides, Harris and Tolias, Kyriakos and Chatzis, Sotirios},
        title     = {A New Dataset for End-to-End Sign Language Translation: The Greek Elementary School Dataset},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
        month     = {October},
        year      = {2023},
        pages     = {1966-1975}
    }

    @inproceedings{panousis2023discover,
        title={DISCOVER: Making Vision Networks Interpretable via Competition and Dissection},
        author={Panousis, Konstantinos P. and Chatzis, Sotirios},
        booktitle = {Advances in Neural Information Processing Systems},
        volume = {36},
        year={2023}
    }
