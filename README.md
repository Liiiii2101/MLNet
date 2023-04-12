# MLNet
An Automated Deep Learning Pipeline for EMVI and Response Prediction of Rectal Cancer using Baseline MRIs
Paper Link: to be updated
Graph Abstract:
![](https://github.com/Liiiii2101/MLNet/blob/main/graph_abstract.pdf)


# Installation

in your favorite virtual environment:

```bash
pip install -r requirements.txt
```

# Training

Firstly, you have to run a nnunet link: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1

Then, extract stage features (you can use multifeature_extractor.py)

Then, you need to have a csv file where you have your classficaton fold information and id. Also, you need to have a label csv file where you store your targeted labels

Also, you can finetune hyperparameters in the config file



 
