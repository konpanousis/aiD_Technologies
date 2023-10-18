# aiD project: The Text to Speech Module.

For generating high-quality synthetic speech, we explored, developed and deployed the Tacotron-2 model \cite{tacotron2}.
This constitutes a network comprising a sequence-to-sequence prediction model 
with a WaveNet vocoder to produce speech. This system can be trained directly from 
data without relying on complex feature engineering, and achieves state-of-the-art 
sound quality close to that of natural human speech. For training the model, 
we turned to the well-known Mozilla TTS framework. We trained the model on two different languages, 
English and Greek. At the same time, the considered framework allows for easily training and incorporating
different languages to the system.  

## Acknowledgements
This  research  was supported  by  the  the European Union’s Horizon2020 research and innovation program, under grant agreement  No  872139,  project  aiD.

## References 
[1] https://github.com/mozilla/TTS

