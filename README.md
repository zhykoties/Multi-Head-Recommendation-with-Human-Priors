# Don't Waste It: Guiding Generative Recommenders with Structured Human Priors via Multi-head Decoding
<a target="_blank" href="https://arxiv.org/abs/2511.10492">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://huggingface.co/collections/to-be-released">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-blue?style=flat"></a>

<br>
<span>
<b>Authors:</b> 
<a class="name" target="_blank" href="https://zhykoties.github.io/">Yunkai Zhang<sup>1</sup><sup>2</sup><sup>&dagger;</sup></a>, 
<a class="name" target="_blank" href="https://zhangtemplar.github.io/">Qiang Zhang<sup>1</sup></a>, 
<a class="name" target="_blank" href="https://www.ryanflin.com/">Feng (Ryan) Lin<sup>1</sup></a>, 
<a class="name" target="_blank" href="https://q-rz.github.io/">Ruizhong Qiu<sup>1</sup></a>,
<a class="name" target="_blank" href="linkedin.com/in/hanchao-yu-9a9381a7">Hanchao Yu<sup>1</sup></a>,
<a class="name" target="_blank" href="https://github.com/jiayiliu">Jiayi (Jason) Liu<sup>1</sup></a>, 
<a class="name" target="_blank" href="https://sites.google.com/site/yinglongxia/">Yinglong Xia<sup>1</sup></a>,
<a class="name" target="_blank" href="https://www.linkedin.com/in/zhuoran/">Zhuoran Yu<sup>1</sup></a>,
<a class="name" target="_blank" href="https://zheng80.github.io/">Zeyu Zheng<sup>2</sup></a>, 
<a class="name" target="_blank" href="https://dyang39.github.io/">Diji Yang<sup>3</sup></a>
<br>
<sup>1</sup>Meta AI. 
<sup>2</sup>BAIR, UC Berkeley.
<sup>3</sup>UC Santa Cruz.
<sup>&dagger;</sup>Work done at Meta. 
</span>


## Installation

1. Python 3.11.9. Install packages via `pip3 install -r requirements.txt`.
2. Prepare `Pixel8M`, `MerRec`, and `EB-NeRD` Datasets.
    1. Download the processed interactions and item descriptions from Huggingface (to be released) and put into the dataset folder and the information folder, respectively.
    1. Download `Pixel8M` images from [PixelRec](https://github.com/westlake-repl/PixelRec) and put into the cover folder.
    3. Please note that Interactions and Item Information should be put into two folders like:
        ```bash
        ├── dataset # Store Interactions
        │   ├── eb_nerd_512.parquet
        │   ├── merrec_2000.parquet
        │   └── Pixel8M.parquet
        └── information # Store item information
        │   ├── eb_nerd_512.parquet
        │   ├── merrec_2000.parquet
        │   └── Pixel8M.parquet
        └── cover # Store item images
            └── Pixel
        ``` 
        Here dataset represents **data_path**, and infomation represents **text_path**.
3. Prepare pre-trained LLM models, such as [TinyLlama](https://github.com/jzhang38/TinyLlama), [Baichuan2](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base).
4. All the run scripts are located under the `./reproduce` folder.

## Citation
If you find this repo useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{zhang2025dontwasteitguiding,
      title={Don't Waste It: Guiding Generative Recommenders with Structured Human Priors via Multi-head Decoding}, 
      author={Yunkai Zhang and Qiang Zhang and Feng and Lin and Ruizhong Qiu and Hanchao Yu and Jason Liu and Yinglong Xia and Zhuoran Yu and Zeyu Zheng and Diji Yang},
      journal={arXiv preprint arXiv:2511.10492},
      year={2025}
}
```

## Acknowledgement
We want to thank the HLLM codebase from ByteDance: https://github.com/bytedance/HLLM