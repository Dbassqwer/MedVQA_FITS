
# FITS_Model

This repository contains the core code of "TYPE-AWARE MEDICAL VISUAL QUESTION ANSWERING" in ICASSP 2022.

## Setting

### Image Pretrained model
We crawled 727 radiology medical images from [PEIR Digital Library](https://peir.path.uab.edu/library/index.php?/category/106) and pretrained these images on [CotNet152](https://github.com/JDAI-CV/CoTNet) as visual feature extract model.

### Language Pretrained model
We used BioM-ELECTRA-Base-SQuAD2 from [Biom-transformers](https://github.com/salrowili/BioM-Transformers) as text feature extract model.

## Citation
    @inproceedings{DBLP:conf/icassp/ZhangTLWZ22,
      author    = {Anda Zhang and
                   Wei Tao and
                   Ziyan Li and
                   Haofen Wang and
                   Wenqiang Zhang},
      title     = {Type-Aware Medical Visual Question Answering},
      booktitle = {{IEEE} International Conference on Acoustics, Speech and Signal Processing,
                   {ICASSP} 2022, Virtual and Singapore, 23-27 May 2022},
      pages     = {4838--4842},
      publisher = {{IEEE}},
      year      = {2022},
      url       = {https://doi.org/10.1109/ICASSP43922.2022.9747087},
      doi       = {10.1109/ICASSP43922.2022.9747087},
      timestamp = {Fri, 24 Jun 2022 12:17:37 +0200},
      biburl    = {https://dblp.org/rec/conf/icassp/ZhangTLWZ22.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }