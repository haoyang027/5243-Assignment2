# Band Gap Prediction Project

This project aims to predict the band gap of compounds based on various descriptors such as composition, density, and formation energy. The project utilizes different neural network architectures, including a shallow Multi-Layer Perceptron (MLP), a Graph Neural Network (GNN), and a Generative Adversarial Network (GAN), to explore the relationships between the descriptors and the band gap.

## Project Structure

The project is organized into the following directories and files:

- **data/**: Contains the dataset used for training and testing the models.
  - `Bandgap_data.csv`: The dataset includes descriptors and the target variable (band gap).

- **models/**: Contains the model definitions.
  - `mlp_model.py`: Implements a shallow MLP for regression tasks.
  - `gnn_model.py`: Implements a GNN for predicting band gaps based on structural properties.
  - `gan_model.py`: Implements a GAN for generating synthetic data or augmenting the dataset.

- **notebooks/**: Contains Jupyter notebooks for interactive exploration.
  - `bandgap_prediction.ipynb`: Main notebook for data loading, preprocessing, model training, evaluation, and visualization.

- **src/**: Contains source code for data processing, training, evaluation, and visualization.
  - `data_preprocessing.py`: Functions for preprocessing the dataset.
  - `train.py`: Logic for training the models.
  - `evaluate.py`: Functions for evaluating model performance.
  - `visualize.py`: Functions for visualizing results.

- **requirements.txt**: Lists the dependencies required for the project, including TensorFlow, Pandas, NumPy, and Matplotlib.

- **README.md**: Documentation for the project.

- **.gitignore**: Specifies files and directories to be ignored by version control.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd bandgap-prediction-project
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Use the `data_preprocessing.py` script to preprocess the dataset. This includes feature selection and normalization.
   
2. **Model Training**: Run the `train.py` script to train the selected model (MLP, GNN, or GAN) on the preprocessed data.

3. **Model Evaluation**: Use the `evaluate.py` script to evaluate the trained model's performance using metrics such as Mean Squared Error (MSE) and R^2 score.

4. **Visualization**: Utilize the `visualize.py` script to generate plots comparing predicted vs. actual band gaps.

5. **Interactive Exploration**: Open the `bandgap_prediction.ipynb` notebook for an interactive interface to explore the models and their performance.

## Notes

- The choice of model architecture (MLP, GNN, GAN) depends on the specific characteristics of the dataset and the desired outcomes.
- The project can be extended by incorporating additional features or experimenting with different model architectures and hyperparameters.
- Ensure that the dataset is properly formatted and cleaned before running the preprocessing and training scripts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.