# WAN
This repository provides an implementation of the ideation about self-harm and suicide prediction model (ISSPM) introduced in paper "Network-based prediction of the disclosure of ideation about self-harm and suicide in online counseling sessions".

ISSPM is a novel network-based prediction model harnessing the strength of complex network theory. It predicts the expression of self-harm and suicide ideation in an online text-based counseling platform. With such assistance, counselors can prepare in advance to pay attention to the impending utterance of ideation about self-harm and suicide.

The general idea of ISSPM is showed in the figure below:

![figure](https://github.com/zhongzhixu/WAN/blob/main/fig/figure2-01.jpg)
Figure 1. The network-based transition point prediction method. (a) A subgraph of the word affinity network. Solid lines represent relationships of word co-occurrence, dash lines represent relationships of word similarity; (b) One-tail z-test; (c)-(d) S-scores for two modules that are topologically closer or topologically distant. (e) Footprints of a block. Words within a block tend to form connected subgraphs, or block module. 

The data is provided by HKU and OpenUp. The data can not be made available to others according to the University's ethical approval.

![figure](https://github.com/zhongzhixu/WAN/blob/main/fig/figure3.png)

This figure examplifies the distance between blocks in WAN. Red module represents ideation about self-harm and suicide block (ISSB),Yellow module stands for prior ISSB block (PISSB), Blue module is a non-ISS block (NISSB). In this example, ISSB is closer to PISSB than to NISSB.

## Environment:
Python 3.6

Spyder 3.1.1

## Files in the folder
DATA: data_sample.csv,
kg_final.txt

data_loader.py,
model.py,
















