# aiD project: The Sign Language Translation Module.

Automating sign language translation (SLT) is a challenging
real-world application. Despite its societal importance,
though, research progress in the field remains rather
poor. Crucially, existing methods that yield viable performance
necessitate the availability of laborious to obtain
gloss sequence groundtruth.
To address the challenge of end-to-end sign language translation, we turned to Transformer-based architectures,
and particularly the Sign Language Transformer (SLT) paradigm. 
Our most important goal was to devise an end-to-end SLT modeling approach that completely obviates
the need of using SLR groundtruth information (glosses) as part of the model pipeline. 
On this basis, we developed the sLWTA-SLT paradigm [1],
a novel approach for SLT, achieving state-of-the-art results without using any glosses, 
while at the same time  To further enhance the generability of the proposed framework, 
we introduce an additional feature extraction method using AlphaPose[2]; 
this constitutes a well-known engine allowing the extraction of keypoints pertaining to a human pose, including body, 
face and hand keypoints. 

## Acknowledgements
This  research  was supported  by  the  the European Union’s Horizon2020 research and innovation program, under grant agreement  No  872139,  project  aiD.

## References 
[1] Andreas Voskou et al. “Stochastic Transformer Networks With Linear Competing Units: Application To End-
to-End SL Translation”. In: Proc. ICCV. 2021.

[2] Hao-Shu Fang et al. “AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-
Time”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2022)
## Citation

    @InProceedings{Voskou_2021_ICCV,
	    author    = {Voskou, Andreas and Panousis, Konstantinos P. and Kosmopoulos, Dimitrios and Metaxas, Dimitris N. and Chatzis, Sotirios},
	    title     = {Stochastic Transformer Networks With Linear Competing Units: Application To End-to-End SL Translation},
	    booktitle = {Proc. ICCV},
	    year      = {2021},
    }

    @InProceedings{Voskou_2023_ICCV,
        author    = {Voskou, Andreas and Panousis, Konstantinos P. and Partaourides, Harris and Tolias, Kyriakos and Chatzis, Sotirios},
        title     = {A New Dataset for End-to-End Sign Language Translation: The Greek Elementary School Dataset},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
        month     = {October},
        year      = {2023},
        pages     = {1966-1975}
    }