# WAN
This repository provides an implementation of the ideation about self-harm and suicide prediction model (ISSPM) introduced in paper "Network-based prediction of the disclosure of ideation about self-harm and suicide in online counseling sessions".

ISSPM is a novel network-based prediction model harnessing the strength of complex network theory. It predicts the expression of self-harm and suicide ideation in an online text-based counseling platform. With such assistance, counselors can prepare in advance to pay attention to the impending utterance of ideation about self-harm and suicide.

The general idea of ISSPM is showed in the figure below:

![figure](https://github.com/zhongzhixu/WAN/blob/main/fig/figure2-01.jpg)
Figure 1. The network-based transition point prediction method. (a) A subgraph of the word affinity network. Solid lines represent relationships of word co-occurrence, dash lines represent relationships of word similarity; (b) One-tail z-test; (c)-(d) S-scores for two modules that are topologically closer or topologically distant. (e) Footprints of a block. Words within a block tend to form connected subgraphs, or block module. 

The data is provided by HKU and OpenUp. The full dataset can not be made publicly available according to the University's ethical approval. Instead, we uploaded two fictitious examples in this repository for readers to test the code. The full data can be accessed upon request and a signment relevant data usage protocal.

![figure](https://github.com/zhongzhixu/WAN/blob/main/fig/figure3.png)

This figure examplifies the distance between blocks in WAN. Red module represents ideation about self-harm and suicide block (ISSB),Yellow module stands for prior ISSB block (PISSB), Blue module is a non-ISS block (NISSB). In this example, ISSB is closer to PISSB than to NISSB.

## Environment:
Python 3.6

Spyder 3.1.1

## Files in the folder
DATA: 

two_fictitious_conversation_examples.xlsx: This file contains two examplary and fictitious chats. Trials can be carried out using the two examples.

confirm_risky_blocks.csv: Blocks with high risk of Self-harm and Suicidal Ideation (SSI)

output_distance.csv：The distribution of NSSI-SSI and PSSI-SSI distances.

viz.csv: for visualization.

CODE:

1graph_distancing_preceding_vs_random.py

2visualize.py

@@functions.py: This contains functions needed in the main program.

@@keywords.py: This is to extract SSI keywords.
















