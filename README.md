# Tabular-Data-Transfer-Learning-implementation
Implementing this TDTL research paper for CIBMTR - Equity in post-HCT Survival Predictions competiton dataset
### Approaching the competition for the CIBMTR - Equity in post-HCT Survival Predictions, it was clear most of the attempted solutions lacked in feature variety because simple machine learning models were vastly outperforming NNs. Additionally, most top solutions for the competition involved some forms of feature engineering.So, I came up with the idea of augmenting the data using the conversion of tabular data into images. Essentially the idea boils down to the fact that NNs have a famously bad time learning and training on tabular data not only because of the usual small size of such datasets but also because of lack of homogenity of these datasets. So, instead of bruteforcing NNs to learn from tabular data, a better approach would be converting tabular data into images using statistical scaling methods. Following which, the pre trained NN on other images can draw insights from the new converted dataset of images. So I decided to implement this research paper (by Bragilovsky et al) that closely matches with my theoretical hypothesis. https://www.sciencedirect.com/science/article/abs/pii/S1568494623007664. The paper additionally suggests using knowledge distillation on tabular to images trained NNs to further cleanse the insights drawn into the data into a useful format. This is my analysis and cross application of the research paper for this dataset/problem. 


### **Methodology**
This notebook combines **traditional machine learning (XGBoost)** with **neural network-based feature engineering** to predict survival probabilities for patients post-HCT. The approach leverages two key innovations from the research paper:  
1. **Tabular-to-Image Conversion**, which transforms raw features into synthetic images for CNN training.  
2. **Iterative FCNN Generations**, a progressive learning framework where each subsequent neural network builds on prior predictions to refine feature extraction.  

---

#### **1. Data Preprocessingg**
**Goal:** Prepare data for both XGBoost and neural networks while avoiding overfitting/leakage.  

- **Categorical Features:**  
  Missing values were imputed with `"NAN"`, and features were factorized into integer codes:  
  \[
  \text{Factorized } c \in \text{Categorical Features} \rightarrow \text{Integer codes}
  \]
- **Numerical Features:**  
  - XGBoost handles `NaN` values directly.  
  - Neural networks require imputation (mean strategy):  
    \[
    \text{Imputed Value} = \frac{1}{n} \sum_{i=1}^{n} \text{Non-NaN Values}
    \]
- **Low-Variance Features:**  
  Removed using `VarianceThreshold` with \( \text{threshold} = 0.01 \):  
  \[
  \text{Variance}(X) \leq \text{threshold} \implies \text{Feature discarded}
  \]

---

#### **2. Feature Importance Calculation**
**Goal:** Identify the most predictive features for image generation.  
- **Pearson Correlation:**  
  Calculated between each feature \( X_i \) and survival probability \( y \):  
  \[
  \rho_{X_i, y} = \frac{\text{Cov}(X_i, y)}{\sigma_{X_i} \sigma_{y}}
  \]
- **Sorting:**  
  Features were reordered by \( |\rho_{X_i, y}| \), prioritizing those with high correlation with \( y \).

---

#### **3. Tabular-to-Image Conversion**
**Goal:** Map tabular data to 2D synthetic images for CNN training.  
- **Grid Creation:**  
  Features were distributed on a \( \lceil \sqrt{k} \rceil \times \lceil \sqrt{k} \rceil \) grid (\( k \) = number of features).  
- **Feature Placement:**  
  For each feature \( i \):  
  \[
  r = \left\lfloor \sqrt{i} \right\rfloor, \quad c = i \% \text{cols}
  \]
  \[
  \text{if } r > \text{rows} - 1 \implies r \leftarrow r - 1, \quad c \leftarrow c + 1
  \]
- **Normalization:**  
  Images were scaled to \( [0, 1] \), resized to \( 64 \times 64 \), and converted to RGB for CNN compatibility.

  ---

#### **4. Teacher Model (CNN)**
**Goal:** Learn survival patterns from synthetic images using a pre-trained CNN.  
- **Architecture:**  
  MobileNetV2 with pre-trained weights loaded locally:  
  \[
  \text{MobileNetV2} \rightarrow \text{Dense Layers} \rightarrow \text{Survival Probability}
  \]
- **Training:**  
  Trained with \( \text{MSE loss} \) on augmented images to improve robustness.

---

#### **5. Iterative FCNN Generations**
**Goal:** Refine feature extraction through progressive learning.  
- **Generations:**  
  Trained \( G = 3 \) FCNNs sequentially:  
  \[
  \text{Generation}_g \text{ learns from } \text{Generation}_{g-1} \text{ predictions}
  \]
- **Architecture:**  
  Each FCNN had:  
  \[
  \text{Input} \rightarrow \text{Dense(256)} \rightarrow \text{Dropout(0.5)} \rightarrow \text{Dense(128)} \rightarrow \text{Dense(64)} \rightarrow \text{Output}
  \]
- **Loss Function:**  
  Trained with \( \text{MSE loss} \):  
  \[
  \mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]

---

#### **6. Distilled Feature Selection**
**Goal:** Retain the most valuable features from iterative FCNN generations.  
- **Cross-Validation:**  
  Trained XGBoost on distilled features across \( 5 \)-folds to compute importance scores.  
- **Selection:**  
  Kept top \( 20\% \) of features with the highest importance:  
  \[
  \text{Selected Features} = \left\{ f \in \text{Features} \mid \text{Importance}(f) > \text{Threshold} \right\}
  \]

---

#### **7. Final XGBoost Model**
**Goal:** Combine original features with distilled features for optimal performance.  
- **Baseline Features:**  
  Used raw tabular data (no imputation for numerical features).  
- **Distilled Features:**  
  Added predictions from all FCNN generations (filtered to top \( 20\% \)).  
- **Hyperparameters:**  
  Original XGBoost parameters remained unchanged:  
  \[
  \text{Parameters} = \left\{
    \begin{array}{l}
      \text{max_depth} = 3, \\
      \text{learning_rate} = 0.02, \\
      \text{device} = \text{cuda}, \\
      \text{enable_categorical} = \text{True}
    \end{array}
  \right.
  \]

---

#### **8. Submission Preparation**
**Goal:** Generate test predictions without data leakage.  
- **Test Preprocessing:**  
  Applied the same transformations as training (imputation, variance filtering).  
- **Iterative FCNN Predictions:**  
  Generated predictions for each generation and concatenated them.  
- **Averaging:**  
  Cross-validated predictions were averaged to reduce variance:  
  \[
  \hat{y}_{\text{final}} = \frac{1}{G} \sum_{g=1}^{G} \hat{y}_{g}
  \]
- **Clipping:**  
  Ensured predictions \( \in [0, 1] \).

---

#### **Key Innovations**
1. **Tabular-to-Image Conversion:**  
   Enabled CNNs to learn spatial patterns from tabular data.  
2. **Iterative FCNN Generations:**  
   A knowledge distillation framework where each model learns from prior predictions:  
   \[
   \text{Generation}_g \text{ input} = \left[ \text{Baseline Features}, \hat{y}_{1}, \hat{y}_{2}, \dots, \hat{y}_{g-1} \right]
   \]
3. **Hybrid Architecture:**  
   Combined CNN’s pattern recognition with XGBoost’s robustness for survival analysis.

---


---

#### **Code Structure**
# 1. Preprocessing → Clean data for XGBoost and neural networks
# 2. Image Generation → Create synthetic images from reordered features
# 3. Teacher Training → Train MobileNetV2 on images
# 4. Iterative FCNNs → Generate predictions across generations
# 5. Feature Selection → Retain top distilled features
# 6. Final XGBoost → Combine features and train with baseline hyperparameters
# 7. Submission → Process test data and average predictions
