## Benchmark models and ensembles for multiple foliar disease classification.

Code for the paper <a href="https://rohit12.com/files/SEMFD_Net.pdf" target="_blank"> SEMFD-Net : A Stacked Ensemble for Multiple Foliar Disease Classification</a>.

<strong> Note </strong>: Complete code will be updated soon. 

<strong>Abstract</strong> : Foliar diseases account for upto 40% to the loss of annual crop yield worldwide. This necessitates early detection of these diseases in order to prevent spread and reduce crop damage. The PlantVillage Dataset is the largest open-access database comprising 38 classes of healthy and diseased leaves. However this dataset contains images of leaves taken in a controlled environment which severely restricts the portability of models trained on this dataset to the real world. Motivated by the need to detect a variety of leaf diseases captured under diverse conditions and backgrounds, as is the case presently where many farmers do not have access to lab infrastructure or high-end cameras, we choose the <a href="https://github.com/pratikkayal/PlantDoc-Dataset" target="_blank"> PlantDoc </a> dataset for our experiments. This dataset contains images comprising a subset of 27 classes of the PlantVillage dataset taken under different backgrounds and of varying resolutions. In this paper, we first present a new set of baselines for foliar disease classification using images taken in the field highlighting the inadequacy of current benchmarks. Secondly, we propose a Stacked Ensemble for Multiple Foliar Disease classification (SEMFD-Net), an ensemble model created by stacking a subset of our baseline models and a simple feed-forward neural network as our meta-learner which significantly outperforms the baselines. Our ensemble model also outperforms two other popular ensembling methods namely plurality voting and averaging. 



<hr />

### Pre-trained models

Uncropped-PlantDoc Dataset

Model | Link
--- | --- 
VGG16 | [link](https://drive.google.com/file/d/1YuUCLv8SmWFtI6Ijjqh_8__VXJukHnwR/view?usp=sharing)
DenseNet121 | [link](https://drive.google.com/file/d/1bKCKqlc3Q_jw4xLD-LUNt74-3DDyPRru/view?usp=sharing)
ResNet50 | [link](https://drive.google.com/file/d/1xMVx--h3w-75lzLOVR9yTjrhZyaNN4FB/view?usp=sharing)
ResNet101 | [link](https://drive.google.com/file/d/18RyxmXORx8pkdzM3YnU11Ulry3qrtqVK/view?usp=sharing)
ResNeSt50 | [link](https://drive.google.com/file/d/198y2cLRoO0YoFAZ4QYhkt_B-VcmT-9nb/view?usp=sharing)
ResNeSt101 | [link](https://drive.google.com/file/d/1ZUDgxX64omJPuQPMEibSr3MC0RQtjoSo/view?usp=sharing)
ViT-B/16 | [link](https://drive.google.com/file/d/1wIWETNHoqM5OqefMMuBsTqogmhzFVuX6/view?usp=sharing)


Cropped-PlantDoc Dataset

Model | Link
--- | --- 
VGG16 | [link](https://drive.google.com/file/d/1lfYdPMCZXyeNIkARp-MblKcIv1aj02iH/view?usp=sharing)
DenseNet121 | [link](https://drive.google.com/file/d/1pjBOq2O9csoNI9j9yBB7JltEoGJBFJlo/view?usp=sharing)
ResNet50 | [link](https://drive.google.com/file/d/1dYBeUPynQjsHFVfocNCRSmN--HyWeEHM/view?usp=sharing)
ResNet101 | [link](https://drive.google.com/file/d/1ztLanWpml3WtxaA7XLFY1xJZ7QjW5ZcO/view?usp=sharing)
ResNeSt50 | [link](https://drive.google.com/file/d/1fd3jA40M-mc39jy514ZuU6epKxRVWblj/view?usp=sharing)
ResNeSt101 | [link](https://drive.google.com/file/d/1FbeJeoNw4civZeoDVxtg_9LkkSDqa7fd/view?usp=sharing)
ViT-B/16 | [link](https://drive.google.com/file/d/1QRbfdcz-VxrlAziY7RrjKvhznh91Uobo/view?usp=sharing)
SEMFD-Net | [link](https://drive.google.com/file/d/1eGmbOfVFWIYWjAPFQDK5arhLrTweygrT/view?usp=sharing)

<hr />