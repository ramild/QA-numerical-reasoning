# numgment

This library allows:

1. Sampling tuples in **ascending order** from arbitrary Cartesian product of discrete sets **exactly uniformly**: ![equation](https://latex.codecogs.com/svg.image?\textbf{x}&space;=&space;(x_1,&space;x_2,&space;\dots,&space;x_n)&space;\sim&space;\text{Uniform}(S_1&space;\times&space;S_2&space;\times&space;\dots&space;\times&space;S_n)). This is done using dynamic programming technique in ![equation](https://latex.codecogs.com/svg.image?O(n&space;\cdot&space;M^2)) time where ![equation](https://latex.codecogs.com/svg.image?M&space;=&space;\max(|S_1|,&space;|S_2|,&space;\dots,&space;|S_n|)).

2. Augmenting Question Answering datasets involving numerical reasoning. This is done by changing the numbers from a (question, passage) pair to some other random numbers of the same magnitude and the same number of trailing zeros (plus some other possible constraints) while maintaining the original order of these numbers. Uniform sampling from step 1 is used for this purpose. Keeping the original order is needed to compute the answer for the augmented sample in the cases where the order of numbers matters in the derivation (```"How many yards longer was the second-longest touchdown compared to the shortest?"```). An example for this use case is provided for the [DROP dataset](https://allennlp.org/drop):  
 ```console
python augment_drop_data/main.py --augmentation_config_path 'augment_drop_data/augmentation_config.json' --train_data_path 'drop_dataset/drop_dataset_train.json' --prediction_train_path 'predictions_on_original_train.json' --output_path 'new_train_data.json'
```