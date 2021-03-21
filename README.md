# Jujeop: Korean Puns for K-pop Stars on Social Media

Jujeop is  a  type  of  pun  and  a  unique  way for  fans  to  express  their  love  for  the  K-popstars  they  follow  using  Korean. 

One  of  the unique  characteristics  of Jujeop is  its  use  of exaggerated expressions to compliment K-pop stars, which contain or lead to humor. 

Based on this characteristic, Jujeop can be separated into four distinct types, with their own lexical collocations: 
 
***(1) Fragmenting words to createa twist, (2) Homophones and Homographs, (3) Repetition, and (4) Nonsense***. 

Thus, the current study first defines the concept of Jujeop in Korean, manually labels 8.6K Jujeop comments and annotates the comments to one of the four Jujeop types. Using the annotated corpus, this study proposes two classifiers; CNN, BiLSTM and additionally KoBERT to verify the definition of the Jujeop comments. We have made our dataset publicly available for future research of Jujeop expressions.


## Jujeop Data Description

The dataset for each result condition can be downloaded by running the file in the dataset directory. All the Jujeop file consist of .txt file type that include title, text, label and	type. Not Jujeop data file is provided as not_jujeop.txt that also includes title, text, label, type. Additionally, we also provide a channel list file as channel.txt that includes youtube query.


### Fragmenting  words  to  create  a  twist
The comments in this type intentionally fragment aspecific word and extract/concentrate a single character from the word to disguise the word’s full meaning (e.g., ‘pretty’ to ‘t’), in order to create a twist in the sentence meaning. The examples are attached as below. 

<p align="center"><img src="https://user-images.githubusercontent.com/18303573/105449577-0278a480-5cbc-11eb-9788-d96a54040703.png" width="500" height="500"></p>


### Homophones and Homographs
Users can employ specific lexical features of homophones and homographs to make a Jujeop comment. After a user makes his/her first sentence with the original meanings of words, they employ other word meanings in the second sentence to compliment the K-pop stars while allowing other users to enjoy the fun.

<p align="center"><img src="https://user-images.githubusercontent.com/18303573/105453743-07415680-5cc4-11eb-81f9-9b06ad066e0f.png" width="500" height="500" class="center"></p>


### Repetition
This is a type of repetition of thesame phrase. As presented in the following example, the comments in this type employ repetition to emphasize the complimentary meanings on the K-pop stars.

<p align="center"><img src="https://user-images.githubusercontent.com/18303573/105449682-35229d00-5cbc-11eb-8829-82c576e5d5a7.png" width="500" height="500" class="center"></p>


### Nonsense
The comments in this type includethe K-pop stars within fictions. The majority of such comments flatter the stars by using exaggerated and almost nonsensical, over the top expressions. Representative examples are presented below:

<p align="center"><img src="https://user-images.githubusercontent.com/18303573/105449717-44a1e600-5cbc-11eb-8b83-d6ab6cfe6c12.png" width="500" height="500" class="center"></p>



## Requirements
* Python >= 3.6
* TensorFlow >= 1.7
* Keras >= 2.1.5

### If you want to implement KoBERT
* Pytorch >= 1.7.0
* transformers==2.1.1
* sentencepiece==0.1.85
* git+https://git@github.com/SKTBrain/KoBERT.git@master
* gluonnlp
* tqdm

## Experiment
We employed deep neural network model to classify Jujeop. Additionally, we employed clustering to check whether each type of the Jujeop comments has similar lingustic dependency relations.

## Experiment Results

### Binary Classification Result (Jujeop vs non-Jujeop)
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Classifier</th>
    <th class="tg-0pky">Class</th>
    <th class="tg-0lax">Precision</th>
    <th class="tg-0lax">Recall</th>
    <th class="tg-0pky">F1-score</th>
    <th class="tg-0lax">Accuracy</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="2">CNN</td>
    <td class="tg-0pky">Jujeop</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal;text-decoration:none">75.41%</span></td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal;text-decoration:none">72.44%</span></td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal;text-decoration:none">73.90%</span></td>
    <td class="tg-0lax" rowspan="2"><span style="font-weight:400;font-style:normal;text-decoration:none">69.05%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">non-Jujeop</td>
    <td class="tg-0lax">60.23.0%</td>
    <td class="tg-0lax">63.86%</td>
    <td class="tg-0pky">61.99%</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="2">BiLSTM</td>
    <td class="tg-0lax">Jujeop</td>
    <td class="tg-0lax">77.59%</td>
    <td class="tg-0lax">72.70%</td>
    <td class="tg-0lax">75.07%</td>
    <td class="tg-0lax" rowspan="2">70.79%</td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:normal;font-style:normal;text-decoration:none">non-Jujeop</span></td>
    <td class="tg-0lax">61.90%</td>
    <td class="tg-0lax">67.87%</td>
    <td class="tg-0lax">64.75%</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="2">KoBERT</td>
    <td class="tg-0lax">Jujeop</td>
    <td class="tg-0lax">80.45%</td>
    <td class="tg-0lax">74.54%</td>
    <td class="tg-0lax">77.38%</td>
    <td class="tg-0lax" rowspan="2">73.65%</td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:normal;font-style:normal;text-decoration:none">non-Jujeop</span></td>
     <td class="tg-0lax">64.98%</td>
    <td class="tg-0lax">72.29%</td>
    <td class="tg-0lax">68.44%</td>
  </tr>
</tbody>
</table>

### Clustering Result Between four types of Jujeop
<p align="center"><img width="360" alt="clustering" src="https://user-images.githubusercontent.com/18303573/111910458-cb98f200-8aa4-11eb-9d9e-a43bb9794b66.png"></p>


&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
