# Jujeop

Jujeop is  a  type  of  pun  and  a  unique  wayfor  fans  to  express  their  love  for  the  K-popstars  they  follow  using  Korean.  One  of  theunique  characteristics  ofJujeopis  its  use  ofexaggerated expressions to compliment K-pop stars, which contain or lead to humor. Basedon this characteristic, Jujeop can be separatedinto four distinct types, with their own lexical collocations: ***(1) Fragmenting words to createa twist, (2) Homophones and Homographs, (3) Repetition, and (4) Nonsense***. Thus, the currentstudy first defines the concept of Jujeop in Korean, manually labels 8.6K Jujeop comments and annotates the comments to one of the four Jujeop types. Using the annotated corpus, this study proposes two classifiers, which are based on deep neural networks, to investigate whether or not each comment is a Jujeop comment. We have made our dataset publicly available for future research of Jujeop expressions.


## Jujeop Data Description
### Fragmenting  words  to  create  a  twist
The comments in this type intentionally fragment aspecific word and extract/concentrate a single character from the word to disguise the word’s full meaning (e.g., ‘pretty’ to ‘t’), in order to create a twist in the sentence meaning. The examples are attached as below. 

(Image)


### Homophones and Homographs
Users can employ specific lexical features of homophones and homographs to make a Jujeop comment. After a user makes his/her first sentence with the original meanings of words, they employ other word meanings in the second sentence to compliment the K-pop stars while allowing other users to enjoy the fun.

(Image)


### Repetition
This is a type of repetition of thesame phrase. As presented in the following example, the comments in this type employ repetition to emphasize the complimentary meanings on the K-pop stars.

(Image)


### Nonsense
The comments in this type includethe K-pop stars within fictions. The majority of such comments flatter the stars by using exaggerated and almost nonsensical, over the top expressions. Representative examples are presented below:

(Image)


The dataset for each result condition can be downloaded by running the file in the dataset directory. All the Jujeop file consist of .txt file type that include title, text, label and	type. Not Jujeop data file is provided as not_jujeop.txt that also includes title, text, label, type. Additionally, we also provide a channel list file as channel.txt that includes youtube query.


## Requirements
* Python >= 3.6
* TensorFlow >= 1.7
* Keras >= 2.1.5
* Pytorch >= 1.7.0

## Clone
```
git clone 
```

## Experiment
We provide deep neural network model to classify Jujeop.

**BiLSTM**
```
python bilstm.py
```
**CNN**
```
python cnn.py
```
**KoBERT**
```
python kobert.py
```

## Experiment Results

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
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal;text-decoration:none">31.98%</span></td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal;text-decoration:none">51.63%</span></td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal;text-decoration:none">39.50%</span></td>
    <td class="tg-0lax" rowspan="2"><span style="font-weight:400;font-style:normal;text-decoration:none">72.02%</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">non-Jujeop</td>
    <td class="tg-0lax">88.03%</td>
    <td class="tg-0lax">76.40%</td>
    <td class="tg-0pky">81.80%</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="2">BiLSTM</td>
    <td class="tg-0lax">Jujeop</td>
    <td class="tg-0lax">40.09%</td>
    <td class="tg-0lax">54.72%</td>
    <td class="tg-0lax">46.28%</td>
    <td class="tg-0lax" rowspan="2">76.65%</td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:normal;font-style:normal;text-decoration:none">non-Jujeop</span></td>
    <td class="tg-0lax">88.89%</td>
    <td class="tg-0lax">81.59%</td>
    <td class="tg-0lax">85.08%</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="2">KoBERT</td>
    <td class="tg-0lax">Jujeop</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax" rowspan="2">1</td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:normal;font-style:normal;text-decoration:none">non-Jujeop</span></td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">1</td>
  </tr>
</tbody>
</table>