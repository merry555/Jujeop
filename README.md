# Jujeop: Korean Puns for K-pop Stars on Social Media

### <a href="https://sites.google.com/site/socialnlp2021/">SocialNLP 2021 @ NAACL DATA Paper</a>

We provide the first definition of the Jujeop comments (주접 댓글) in South Korea and human-annotated 8.6K Jujeop corpus. Jujeop is  a  type  of  pun  and  a  unique  way for  fans  to  express  their  love  for  the  K-pop stars  they  follow  using  Korean. One  of  the unique  characteristics  of Jujeop is  its  use  of exaggerated expressions to compliment K-pop stars, which contain or lead to humor. Based on this characteristic, Jujeop can be separated into four distinct types, with their own lexical collocations: 
 
***(1) Fragmenting words to createa twist, (2) Homophones and Homographs, (3) Repetition, and (4) Nonsense***. 

## Jujeop Data Description

The dataset for each result condition can be downloaded by running the file in the ``dataset`` directory. All the Jujeop file consist of .txt file type that include title, text, label and	type. Not Jujeop data file is provided as ``not_jujeop.txt`` that also includes title, text, label, type. Additionally, we also provide a video channel list file as ``channel.txt`` that includes youtube video query. We uploaded the Youtube crawler ``crawler.py``, we implemented to collect video title, comments, user name, and number of likes. 

| <img src="https://user-images.githubusercontent.com/18303573/105449577-0278a480-5cbc-11eb-9788-d96a54040703.png" alt="" width="400px" height="400px"/> | <img src="https://user-images.githubusercontent.com/18303573/105453743-07415680-5cc4-11eb-81f9-9b06ad066e0f.png" alt="" width="400px" height="400px"/> |
|:--:|:--:| 
| *Fragmenting  words  to  create  a  twist* | *Homophones and Homographs* |
| The comments in this type intentionally fragment aspecific word and extract/concentrate a single character from the word to disguise the word’s full meaning (e.g., ‘pretty’ to ‘t’), in order to create a twist in the sentence meaning. The examples are attached as below. | Users can employ specific lexical features of homophones and homographs to make a Jujeop comment. After a user makes his/her first sentence with the original meanings of words, they employ other word meanings in the second sentence to compliment the K-pop stars while allowing other users to enjoy the fun. |

| <img src="https://user-images.githubusercontent.com/18303573/115512281-c7870c80-a2bc-11eb-9819-d2ca98dd7ab8.png" alt="" width="400px" height="400px"/> | <img src="https://user-images.githubusercontent.com/18303573/115512354-dc63a000-a2bc-11eb-9659-df64f46409ca.png" alt="" width="400px" height="400px"/> | 
|:--:|:--:| 
| *Repetition* | *Nonsense* |
| This is a type of repetition of thesame phrase. As presented in the following example, the comments in this type employ repetition to emphasize the complimentary meanings on the K-pop stars. | The comments in this type includethe K-pop stars within fictions. The majority of such comments flatter the stars by using exaggerated and almost nonsensical, over the top expressions. 

## Experiment
We employed deep neural network models to classify Jujeop for verification of the annotated corpus quality. Within the ``models/binary`` folder we uploaded classification models to binarize comments into Jujeop and non-Jujeop types. Additionally, we conducted multi-class classification for each Jujeop type which uploaded in ``models/multiclass`` folder. We're always welcome to get feedback for improving model performance! 😊

### Requirements
* Python >= 3.6
* TensorFlow >= 1.7
* Keras >= 2.1.5

### If you want to implement KoBERT
* Pytorch >= 1.7.0
* transformers >= 3.5.0
* sentencepiece==0.1.85
* MXNet >= 1.4.0
* onnxruntime >= 0.3.0
* git+https://git@github.com/SKTBrain/KoBERT.git@master
* gluonnlp
* tqdm

## Contributors 
<a href="https://sori424.github.io/">Soyoung Oh*</a> 🥰 <a href="https://sites.google.com/view/jisukim8873/home">Jisu Kim*</a> 😸 <a href="https://sites.google.com/view/leepeel">Seunpeel Lee*</a> 👓 

Corresponding author (Professor) <a href="http://eunilpark.com"> Eunil Park :sunglasses: </a> 

*: Equal Contribution

## References


<!---
## Experiment Results
--->

<!---
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
--->

<!---
### Clustering Result Between four types of Jujeop
<p align="center"><img width="360" alt="clustering" src="https://user-images.githubusercontent.com/47997074/114311410-945cb480-9b29-11eb-8123-c7ec41711e27.png"></p>
--->

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
