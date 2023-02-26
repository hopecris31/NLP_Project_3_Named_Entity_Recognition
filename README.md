# Named Entity Recognition
## Using the Viterbi Algorithm in a Hidden Markov Model and a Maximum Entropy Markov Model

(These classification tasks were performed on the CoNLL Spanish Corpus)
###### The possible asigned tags include:

B --> Beginning   
I --> Inside
###### Plus one of the following:    
PER --> Person    
LOC --> Location    
ORG --> Organization    
MISC --> Miscellaneous Named Entities    
O --> Non-Entity    

For example, the city "La Coruña" gets the following tags:  
La: B-LOC   
Coruña: I-LOC    

## Hidden Markov Model
The file ner.py implements the Viterby algorithm in a Hidden Markov Model.  The classifier assigns labels
to words based on the probability:     
<img width="86" alt="Screenshot 2023-02-26 at 1 06 17 PM" src="https://user-images.githubusercontent.com/84686517/221428221-f2e5b41f-3668-4326-9d9d-ca6997124303.png">    
That is the probability of assigning a label *ti* to the word *wi* given the word’s feature representation. This
features representation can include information about wi itself as well as information about wi’s neighbors.


## Maximum Entropy Markov Model
This classifier added onto the HMM version, so that it is trained to discriminate based on the following
probability:     
<img width="121" alt="Screenshot 2023-02-26 at 1 05 14 PM" src="https://user-images.githubusercontent.com/84686517/221428165-14ead728-7d69-4006-987c-8866834af083.png">      
which is then expanded to:         
<img width="249" alt="Screenshot 2023-02-26 at 1 03 57 PM" src="https://user-images.githubusercontent.com/84686517/221428074-9e4ce28e-f07c-45d3-8fdd-b529bf3fa14c.png">    
It takes into consideration the predicted tag of the previous word, in addition to the features of the current and neighboring words.
Instead of multiplying together the transition and observation probabilities of the HMM, we are now multiplying together the probabilities predicted by the classifier. 
