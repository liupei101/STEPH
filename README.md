# STEPH: Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in Whole-Slide Image Prognosis

[[Paper]](https://arxiv.org/abs/2603.10526) | [[Acknowledgements]](https://github.com/liupei101/STEPH?tab=readme-ov-file#acknowledgements) | [[Citation]](https://github.com/liupei101/STEPH?tab=readme-ov-file#-citation)

**Abstract**: Whole-Slide Images (WSIs) are widely used for estimating the prognosis of cancer patients. Current studies generally follow a cancer-specific learning paradigm. However, the available training samples for one cancer type are usually scarce in pathology. Consequently, the model often struggles to learn generalizable knowledge, thus performing worse on the tumor samples with inherent high heterogeneity. Although multi-cancer joint learning and knowledge transfer approaches have been explored recently to address it, they either rely on large-scale joint training or extensive inference across multiple models, posing new challenges in computational efficiency. To this end, this paper proposes a new scheme, Sparse Task Vector Mixup with Hypernetworks (STEPH). Unlike previous ones, it efficiently absorbs generalizable knowledge from other cancers for the target via model merging: i) applying task vector mixup to each source-target pair and then ii) sparsely aggregating task vector mixtures to obtain an improved target model, driven by hypernetworks. Extensive experiments on 13 cancer datasets show that STEPH improves over cancer-specific learning and an existing knowledge transfer baseline by 5.14% and 2.01%, respectively. Moreover, it is a more efficient solution for learning prognostic knowledge from other cancers, without requiring large-scale joint training or extensive multi-model inference.

---

We are preparing the camera-ready paper & code & training logs & models and will make them publicly available once finished.

*On updating. Stay tuned.*

📚 Recent updates:
- 26/03/11: initialized a repo for STEPH
- 26/02/20: STEPH is accepted to CVPR 2026

## UNI2-h-DSS Dataset

HF Dataset (with complete DSS labels): https://huggingface.co/datasets/yuukilp/UNI2-h-DSS

## Acknowledgements

We thank the following great works that contribute to this work:

- [UNI](https://github.com/mahmoodlab/UNI): a state-of-the-art foundation model for pathology; it is used to extract patch features from WSIs.
- [UNI2-h features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features): the datasets for this study are derived from it.
- [TCGA GDC Data portal](https://portal.gdc.cancer.gov/): it provides the source data for analysis.

## 📝 Citation

If you find this work helps your research, please consider citing our paper:
```txt
@inproceedings{liu2026steph,
    title={Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in Whole-Slide Image Prognosis}, 
    author={Pei Liu and Xiangxiang Zeng and Tengfei Ma and Yucheng Xing and Xuanbai Ren and Yiping Liu},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={1--12},
    year={2026}
}
```
