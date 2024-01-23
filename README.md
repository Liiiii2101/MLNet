# MLNet🔍
An Automated Deep Learning Pipeline for EMVI and Response Prediction of Rectal Cancer Using Baseline MRI

Paper Link: https://www.nature.com/articles/s41698-024-00516-x


<img src="https://github.com/Liiiii2101/MLNet/blob/main/graphic_abstract.jpg" width="400" />



# Installation💻

in your favourite virtual environment:

```bash
pip install -r requirements.txt
```

# Training🚀

Firstly, you have to run a nnunet link: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1

Then, extract stage features (you can use multifeature_extractor.py)

Then, you need to have a csv file where you have your classification fold information and id. Also, you need to have a label csv file where you store your targeted labels

Also, you can finetune hyperparameters in the config file


Then you can Train with the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python trainer.py --load_json config/nnunet.json 
```
# Evaluation⚡️
TEST with:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_test.py 
```
