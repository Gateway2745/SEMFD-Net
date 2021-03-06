## Stacked Ensemble for Multiple Foliar Disease Classification

Code for the paper <a  href="https://dl.acm.org/doi/10.1145/3493700.3493719"  target="_blank"> SEMFD-Net : A Stacked Ensemble for Multiple Foliar Disease Classification</a>.

Paper Accepted at the Applied Data Science Track, ACM India Joint International Conference on Data Science and Management of Data (CODS-COMAD 2022).

<strong>Abstract</strong> : Foliar diseases account for upto 40% to the loss of annual crop yield worldwide. This necessitates early detection of these diseases in order to prevent spread and reduce crop damage. The PlantVillage Dataset is the largest open-access database comprising 38 classes of healthy and diseased leaves. However this dataset contains images of leaves taken in a controlled environment which severely restricts the portability of models trained on this dataset to the real world. Motivated by the need to detect a variety of leaf diseases captured under diverse conditions and backgrounds, as is the case presently where many farmers do not have access to lab infrastructure or high-end cameras, we choose the  PlantDoc dataset for our experiments. This dataset contains images comprising a subset of 27 classes of the PlantVillage dataset taken under different backgrounds and of varying resolutions. In this paper, we first present a new set of baselines for foliar disease classification using images taken in the field highlighting the inadequacy of current benchmarks. Secondly, we propose a Stacked Ensemble for Multiple Foliar Disease classification (SEMFD-Net), an ensemble model created by stacking a subset of our baseline models and a simple feed-forward neural network as our meta-learner which significantly outperforms the baselines. Our ensemble model also outperforms two other popular ensembling methods namely plurality voting and averaging.

<strong> Dataset Credits </strong>: The Uncropped PlantDoc dataset is available <a  href="https://github.com/pratikkayal/PlantDoc-Dataset"  target="_blank"> here </a> and we generated the Cropped PlantDoc dataset using the annotations available <a  href="https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset"  target="_blank"> here </a>.  

<hr />

Dataset - <br /> 
Our dataset split is available [here](https://drive.google.com/drive/folders/1t4rr_37u5KxBu89ZX-4kth8p-zBsJoZ7?usp=sharing). <br /> 

Setup package - <br /> 
`pip install -e .` <br /> 

To train run - <br /> 
`python run.py configs/config.yaml`

SEMFD-Net Inference - <br />
Download `ensemble.onnx` from [here](https://drive.google.com/file/d/1qNKH2KThRuGfKEckCpwucw-pwQCbfQCE/view?usp=sharing). <br /> 
Refer notebook `demo/ensemble_demo.ipynb`

