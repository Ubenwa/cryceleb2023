## CryCeleb2023 SpeechBrain fine-tuning

All you need to start fine-tuning [SpeechBrain](https://speechbrain.readthedocs.io/) models using the [Ubenwa CryCeleb dataset](https://huggingface.co/datasets/Ubenwa/CryCeleb2023)!

We hope it will help you get to the next level in [CryCeleb2023 challenge](https://huggingface.co/spaces/competitions/CryCeleb2023)

It also reproduces the [official baseline](https://huggingface.co/Ubenwa/ecapa-voxceleb-ft2-cryceleb) model training

`train.ipynb` <a target="_blank" href="https://colab.research.google.com/github/Ubenwa/cryceleb2023/blob/main/train.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> - main code for data preparation and fine-tuning with various configs

`evaluate.ipynb` <a target="_blank" href="https://colab.research.google.com/github/Ubenwa/cryceleb2023/blob/main/evaluate.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> - example of scoring with pre-trained and fine-tuned model

Note that default configurations are optimized for speed and simplicity rather than accuracy

## Cite

### CryCeleb

```bibtex
@article{ubenwa2023cryceleb,
      title={CryCeleb: A Speaker Verification Dataset Based on Infant Cry Sounds},
      author={David Budaghyan and Charles C. Onu and Arsenii Gorin and Cem Subakan and Doina Precup},
      year={2023},
      journal={preprint arXiv:2305.00969},
}
```

### SpeechBrain

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
