# Supervised learning based essay grading

For comparison with LLMs, we used feature-based supervised learning to grade language structure and correctness related categories of the rubric. The labeled training data comes from trial e-exams in Estonian language held in spring 2024. The 9th-grade dataset contains linguistic features extracted from 781 essays, the 12th-grade dataset is based on 764 essays. Each text has been graded by two experts. Our aim was to predict average subscores of individual rubric aspects.

### Features

Using NLP tools, we calculated 122 quantitative linguistic features of the essays related to five aspects: vocabulary, syntax,
punctuation, orthography and morphology, and structuring and formatting. The features include grammatical features (67), lexical features (20), surface features that characterize text complexity (5) or paragraph structure (7), and error features obtained with spelling and grammar correction tools (23).

To generate the datasets, we largely reused the [scripts](https://github.com/tlu-dt-nlp/Estonian-CEFR-Assessment/tree/main/Feature_extraction) developed for predicting the proficiency level of Estonian L2 writings. However, we added measures of paragraph count and length based on tokenization performed with Stanza (version 1.3.0). We also applied a different grammatical error correction model (available [here](https://huggingface.co/tartuNLP/Llammas-base-p1-llama-errors-p2-GEC) and rule-based filtering of correction types.

For the scoring experiments, we associated each feature with one or several of the graded aspects. The features are described in `feat_descriptions.txt`. 

### Regression

We built separate machine learning pipelines for each graded aspect and used 10-fold cross-validation to evaluate their scoring performance. The pipelines use various regression algorithms and feature sets. We treat the average subscore of the two graders as an interval scale of 0 to 3 with distances of 0.5. The scoring pipelines contain the following tasks: standardization of training data, feature selection, training a regression model, and validation of the model on test data. The evaluation is based on the mean absolute error (MAE).

The `regressor_comparison.py` script can be used to combine different regression methods with a varying number of features and define the best parameters for essay scoring. `regressor_validation.py` offers a more detailed overview of the performance of predetermined regression parameters. The feature sets can be imported from `feat_lists_9th_grade.py` and `feat_lists_12th_grade.py`. Some features that occur in the datasets have been removed from the lists to reduce multicollinearity. We identified features that relate to the same grading aspect and have an absolute correlation above 0.8, considering the mean Pearson correlation coefficient across the training folds. From each set of highly correlated features, we kept the feature that had the strongest mean correlation with the target score.

The Pandas, Numpy, and Scikit-learn library are required to run the scripts. In our experiments, we used the Pandas version 1.4.2, Numpy version 1.22.3, and Scikit-learn version 1.0.2.