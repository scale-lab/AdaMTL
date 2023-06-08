# AdaMTL: Adaptive Input-dependent Inference for Efficient Multi-Task Learning

## Introduction

This is the official implementation of the paper: **[AdaMTL: Adaptive Input-dependent Inference for Efficient Multi-Task Learning](https://arxiv.org/abs/2304.08594)**. 

This repository provides a Python-based implementation of the adaptive multi-task learning (MTL) approach proposed in the paper.  Our method is designed to improve efficiency in multi-task learning by adapting inference based on input, reducing computational requirements and improving performance across multiple tasks. The repository is based upon [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and uses some modules from [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch).


## How to Run

To run the AdaMTL code, follow these steps:

1. **Clone the repository**

    ```bash
    git clone https://github.com/scale-lab/AdaMTL.git
    cd AdaMTL
    ```

2. **Install the prerequisites**

    ```bash
    conda env create -f environment.yml
    conda activate adamtl
    ```

3. **Run the code**

    **Stage 1: Training the backbone:**
        ```
        python main.py --cfg configs/swin/<swin variant>.yaml --pascal <path to pascal database> --tasks semseg,normals,sal,human_parts --batch-size <batch size> --ckpt-freq=20 --epoch=1000 --resume-backbone <path to swin weights>
        ```
    
    **Stage 2: Controller pretraining:**
        ```
        python main.py --cfg configs/ada_swin/<swin variant>_<tag/taw>_pretrain.yaml --pascal <path to pascal database> --tasks semseg,normals,sal,human_parts --batch-size <batch size> --ckpt-freq=20 --epoch=100 --resume <path to the weights generated from Stage 1>
        ```
        
    **Stage 3: MTL model training:**
        ```
        python main.py --cfg configs/ada_swin/<swin variant>_<tag/taw>.yaml --pascal <path to pascal database> --tasks semseg,normals,sal,human_parts --batch-size <batch size> --ckpt-freq=20 --epoch=100 --resume <path to the weights generated from Stage 2>
        ```
  
## Authorship
Since the release commit is squashed, the GitHub contributers tab doesn't reflect the authors' contributions. The following authors contributed equally to this codebase:
- [Marina Neseem](https://github.com/marina-neseem)
- [Ahmed Agiza](https://github.com/ahmed-agiza)

## Citation
If you find AdaMTL helpful in your research, please cite our paper:
```
@inproceedings{neseem2023adamtl,
  title={AdaMTL: Adaptive Input-dependent Inference for Efficient Multi-Task Learning},
  author={Neseem, Marina and Agiza, Ahmed and Reda, Sherief},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4729--4738},
  year={2023}
}
```

## License
MIT License. See [LICENSE](LICENSE) file
