# MGD-1k Dataset Documentation

## 1. Dataset Name
MGD-1k: Infrared Meibomian Gland Dataset

## 2. Overview
The MGD-1k dataset contains **1,000 infrared meibography images** of Meibomian Glands.  
All images are annotated by trained investigators under the supervision of MGD experts and ophthalmologists.

The dataset includes:
1. 1000 meibography images  
2. 1000 gland segmentation masks  
3. 1000 eyelid segmentation masks  
4. 6 rounds of meiboscore grading per image  

This makes MGD-1k suitable for segmentation, morphology analysis, and automated MGD grading.

---

## 3. Dataset Structure (In This Repository)

```
dataset/raw/
    images/
    masks_gland/
    masks_eyelid/
    metadata/
```

**Important:**  
Raw dataset files (images & masks) are **not included** here.  
Place your local MGD-1k dataset into this folder following the above structure.

---

## 4. Dataset Distribution

- **Patients:** 320  
- **Age:**  
  - Men: 51 ± 19  
  - Women: 55 ± 19  
- **Sex ratio:**  
  - Men: 322 (32.2%)  
  - Women: 678 (67.8%)  
- **Total images:** 1000  
  - Upper eyelid: 467  
  - Lower eyelid: 533  
- **Gradability:**  
  - 941 fully gradable  
  - 59 ungradable in ≥1 round  
- **Color:** Grayscale  
- **Device:** LipiView II Ocular Surface Interferometer  
- **Collection period:** April 2019 – April 2020

---

## 5. Demographic Insights
MGD-1k includes a diverse population, enabling analysis of age-related MGD severity.  
The male–female distribution reflects real-world clinical sampling.

---

## 6. Dataset Curation & Annotation Pipeline

1. Image acquisition using LipiView II  
2. Manual annotation of:
   - Meibomian glands  
   - Eyelid structures  
3. Verification by MGD experts  
4. Six rounds of meiboscore grading  
5. Gradability filtering  
6. Final dataset:
   - 1000 total images  
   - 941 with complete meiboscores  

This pipeline ensures high-quality pixel-level annotations.

---

## 7. Meiboscore Expert Validation
- Six independent grading rounds  
- High inter-round consistency  
- Strong ground truth for gland dropout quantification  
- Reliable for AI-based meiboscore prediction  

---

## 8. Morphological Coverage
Dataset includes:
- Normal glands  
- Partially atrophic glands  
- Severely atrophic glands  
- Distorted or irregular glands  

Useful for:
- Structure-aware segmentation  
- Morphology-based dropout quantification  
- Biomarker development  

---

## 9. Purpose in This Repository
MGD-1k is used here for:
1. Dual-branch segmentation (eyelid + glands)  
2. DRAMNetUniversal training and ablations  
3. Morphological dropout severity analysis  
4. Future meiboscore prediction models  

---

## 10. Citation

**Reference Paper:**  
Saha et al., *Automated quantification of meibomian gland dropout in infrared meibography using deep learning*, The Ocular Surface, 2022.

**BibTeX:**
```bibtex
@article{saha2022automated,
  title={Automated quantification of meibomian gland dropout in infrared meibography using deep learning},
  author={Saha, Ripon Kumar and Chowdhury, AM Mahmud and Na, Kyung-Sun and Hwang, Gyu Deok and Eom, Youngsub and Kim, Jaeyoung and Jeon, Hae-Gon and Hwang, Ho Sik and Chung, Euiheon},
  journal={The Ocular Surface},
  volume={26},
  pages={283--294},
  year={2022},
  publisher={Elsevier}
}
```

---

## 11. Contact
Dataset inquiries: **ripon.ece@gmail.com**

---

## 12. Note
This repository does **not** include the raw dataset.  
Please place your downloaded dataset files under:

```
dataset/raw/
```

following the structure shown above.
