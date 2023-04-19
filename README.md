# GWL
"A Generalized Weighted Loss for SVC and MLP"

https://arxiv.org/pdf/2302.12011.pdf


The main file for generalized SVC loss is:

 S.cpp

 compiled by 'make'

 Set the chosen generalization scheme (form 1 to 6) in the respecive line 180<->185 of S.cpp .
 Call it with the following arguments:
 ./S dataset-name nFeatures
 e.g.: 
 ./S iono.txt.shuf 34
 ./S sonar.txt.shuf 60
 ./S breast.txt.shuf 9
 ./S german.txt.shuf 24
