'''
This is the anaconda remade as a python file
The main difference is that I used a lot of helper functions instead of writing all the code as is.
Feels more organized to me... If I'd have to do it again, I'd probably use classes instead of having so many functions
'''
###############################################################################################
# IMPORTS
###############################################################################################

# main libraries
import sklearn
import pandas as pd
import seaborn as sns 
import numpy  as np
import matplotlib.pyplot as plt

from IPython.display import display     # to improve df prints

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# kfold cross validation
from sklearn.pipeline import Pipeline   # Allows you to sequentially apply a list of transformers to preprocess the data
from sklearn.model_selection import GridSearchCV    
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# final model evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score

###############################################################################################
# HyperParams
###############################################################################################

RANDOM = 2025
SCALER = StandardScaler()


###############################################################################################
# Helper functions
###############################################################################################

# Prints some info about the dataframe, and it's k first rows.
def print_df_info(df: pd.DataFrame, num_of_rows: int = 5, pre_space: int = 0, post_space: int = 0) -> None:
    print('\n' * pre_space, end='')
    print(f"Number of rows: {df.shape[0]}\nNumber of columns:{df.shape[1]}")
    display(df.head(num_of_rows))
    print('\n' * post_space, end='')


# Splits dataframe into two sperate dataframes, one for data, second for labels
def split_data_label(df: pd.DataFrame, label_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.drop(label_name, axis=1)
    labels = df[label_name]
    return data, labels


# Bar chart for class distribution
# df <- single column df, representing the labels
def bar_chart(df: pd.Series) -> None:
    """
    I use it to see whether classes are well balanced
    """
    class_counts = df.value_counts().sort_index()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Wine - class frequencies (train set)")
    plt.xlabel("Class label")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.show()


# Pearson correlation heat map to see correlation between pairs of features
# df <- df with all features you want to check
def pearson_heat_map(df: pd.DataFrame) -> None:
    """
    helps spot highly correlated pairs of features
    could use PCA or regularisation if the value is greater than 0.75 or 0.8...
    could drop/combine features 

    total_phenols and flavanoids are highly correlated
    """
    corr = df.corr()    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Feature–feature correlation heat-map")
    plt.tight_layout()
    plt.show()


# IQR box plot
# df <- df with all features you want to check
def IQR_box_plot(df: pd.DataFrame) -> None:
    """
    helps find outliers per feature

    ash and proanthocyanins have the most
    """
    outlier_counts = {}

    fig, axes = plt.subplots(5, 3, figsize=(10, 12))        # 15 axes (13 used)
    axes = axes.flatten()

    for idx, col in enumerate(df.columns):
        ax = axes[idx]
        sns.boxplot(x=df[col], ax=ax, color="#69b3a2")
        ax.set_title(col, fontsize=9)
        # ── Tukey fences ────────────────────────────────────────────────────────────
        q1, q3 = np.percentile(df[col], [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        outlier_counts[col] = n_out
        ax.annotate(f"outliers: {n_out}", xy=(0.02, 0.90), xycoords="axes fraction",
                    fontsize=8, color="red", ha="left")

    plt.tight_layout()
    plt.show()

    print("\nOutlier count dataframe:")
    outlier_df = (pd.Series(outlier_counts).sort_values(ascending=False).reset_index().rename(columns={"index": "feature", 0: "n_outliers"}))
    display(outlier_df)
    print('\n' * 3)


# PCA Scatterplot
def PCA_scatter(data: pd.DataFrame, labels: pd.Series) -> None:
    '''
    Used to gauge class separability

    After seeing the pca scatter plot we can make some conclusions
    The three classes are generally well clustered and seperated(with few outliers)
    PC1 is at 35.60%
    PC2 is at 20.00%
    PC1 greatly separates classes 0 and 2
    PC2 separates class 1 from classes 0 and 2
    In total we get that PC1-2 account for 55.6% of the variation(remaining 44.40% being of PC3-13), which is not ideal
    '''
    train_data_scaled = SCALER.fit_transform(data)
    pca = PCA(n_components=2, random_state=RANDOM)
    pc = pca.fit_transform(train_data_scaled)

    pc_df = pd.DataFrame(pc, columns=["PC1", "PC2"]).assign(target=labels)

    fig, ax = plt.subplots(figsize=(8,6))
    for cls in sorted(pc_df['target'].unique()):
        sel = pc_df['target'] == cls
        ax.scatter(pc_df.loc[sel, 'PC1'], pc_df.loc[sel, 'PC2'], label=f"Class {cls}", alpha=0.7)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Wine - first two principal components")
    ax.legend()
    plt.tight_layout()
    plt.show()


# PCA Variance Plot
def PCA_variance(data: pd.DataFrame) -> None:
    '''
    helps choose dimensionality for Feature Engineering
    I'll probably start with the top 8 PCs for 90% var for a baseline and compare to the full 13-feature model

    Got a lot of PC's that fall under very low % values, dropping them can be benifical since they probably just add noise
    Choosing number of PC's
    Number of PCs needed for ≥75 % variance: 5
    Number of PCs needed for ≥80 % variance: 5
    Number of PCs needed for ≥85 % variance: 6
    Number of PCs needed for ≥90 % variance: 8
    '''
    train_data_scaled = SCALER.fit_transform(data)
    pca_full = PCA(random_state=RANDOM).fit(train_data_scaled)

    # Collect variance information
    var_pct  = pca_full.explained_variance_ratio_ * 100     # % variance per PC
    cum_pct  = var_pct.cumsum()                             # cumulative %
    pcs      = np.arange(1, len(var_pct) + 1)


    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(pcs, var_pct, alpha=0.6, label="Individual % variance")

    for bar, pct in zip(bars, var_pct):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,                       # 1 % above the bar
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.plot(pcs, cum_pct, marker="o", color="tab:orange", label="Cumulative % variance")

    # Creates the variance threshold lines 
    ax.axhline(90, ls="--", lw=1, color="gray")
    ax.text(len(pcs), 91, "90 % threshold", ha="right", va="bottom", color="gray")
    ax.axhline(80, ls="--", lw=1, color="gray")
    ax.text(len(pcs), 81, "80 % threshold", ha="right", va="bottom", color="gray")
    ax.axhline(70, ls="--", lw=1, color="gray")
    ax.text(len(pcs), 71, "70 % threshold", ha="right", va="bottom", color="gray")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Percentage of Variance Explained")
    ax.set_title("Wine dataset – Scree & Cumulative Variance")
    ax.set_xticks(pcs)
    ax.set_ylim(0, max(var_pct.max() + 10, 100))
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Help decide how many features to keep
    n75 = np.argmax(cum_pct >= 75) + 1
    print(f"Number of PCs needed for ≥75 % variance: {n75}")
    n80 = np.argmax(cum_pct >= 80) + 1
    print(f"Number of PCs needed for ≥80 % variance: {n80}")
    n85 = np.argmax(cum_pct >= 85) + 1
    print(f"Number of PCs needed for ≥85 % variance: {n85}")
    n90 = np.argmax(cum_pct >= 90) + 1
    print(f"Number of PCs needed for ≥90 % variance: {n90}")
    print('\n' * 3)


# Creates pipeline
def make_pipeline(scaler: any = SCALER, random_state: int = RANDOM) -> Pipeline:
    return Pipeline([
        ("scaler", scaler),
        ("pca",    PCA(random_state=random_state)),
        ("clf",    LogisticRegression(random_state=random_state))
    ])


# defines a grid for CV
def get_param_grid(random_state: int= RANDOM) -> list[dict[str, any]]:
    param_grid = [
        {
            "pca__n_components": [None, 5, 8, 0.90],
            "clf":               [LogisticRegression(max_iter=2000, solver="lbfgs", penalty="l2", random_state=random_state)],
            "clf__C":            [0.01, 0.1, 1, 10],
        },
        {
            "pca__n_components": [None, 5, 8, 0.90],
            "clf":               [LogisticRegression(max_iter=10000,solver="saga", penalty ="l1", random_state=random_state)],
            "clf__C":            [0.01, 0.1, 1, 10],
        },
        {
            "pca__n_components": [None, 5, 8, 0.90],
            "clf":               [RandomForestClassifier(n_jobs=-1,random_state=random_state)],
            "clf__n_estimators": [10, 100, 200, 500, 1000],
            "clf__max_depth":    [None, 5, 10, 20, 40]
        },
        {
            "pca__n_components": [None, 5, 8, 0.90],
            "clf":               [SVC(probability=True, random_state=random_state)],
            "clf__C":            [0.1, 1, 10],
            "clf__gamma":        ['scale', 0.01, 0.1]
        },
        {
            "pca__n_components": [None, 5, 8, 0.90],
            "clf":               [KNeighborsClassifier()],
            "clf__n_neighbors":  [3, 5, 7, 9, 11],
            "clf__weights":      ['uniform', 'distance'],
            "clf__metric":       ['euclidean', 'manhattan']
        },
        {
            "pca__n_components": [None, 5, 8, 0.90],
            "clf":               [GaussianNB()]
        }
    ]
    return param_grid


# Run grid search, uses macro average f1 to score, and 5 fold cross validation 
def run_grid_search(pipe: Pipeline, param_grid: list[dict[str, any]], data: pd.DataFrame, labels: pd.Series) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)
    macro_f1 = make_scorer(f1_score, average="macro")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=macro_f1,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(data, labels)
    return grid


# Prints and displays the results from the CV
def summarize_grid_search(grid: GridSearchCV, top_n: int=10) -> pd.DataFrame:
    # Summary of the results
    cv_results_df = pd.DataFrame(grid.cv_results_)
    sorted_df = cv_results_df.sort_values(by="mean_test_score", ascending=False)    # higher mean test score the better, so we sort by decreasing order
    results_df = sorted_df[["params", "mean_test_score", "std_test_score"]]

    # some formatting to get a better display of the df, gpt's suggestion since i had an issues with cell width being too small
    pd.set_option("display.max_colwidth", None)     # show entire cell contents
    pd.set_option("display.width", None)            # let Jupyter decide total width

    print("\nTop permutations (sorted by mean macro-F1 over 5 folds):")
    display(results_df.head(top_n))

    print("\nBEST permutation:")
    print(f"Macro-F1 = {grid.best_score_:.3f}")
    print(f"Params   = {grid.best_params_}")

    return results_df


# General model evaluation, the function prints out the results
def print_model_evaluation(test_labels: pd.DataFrame, labels_pred: pd.Series) -> None:
    print('\nFirst 5 predictions:')
    for _ in range(5):
        print(f'Example no.{_}, Expected:{test_labels.iloc[_]}, predicted:{labels_pred[_]}')
    print('\n' *2)
    print(f"Accuracy = {accuracy_score(test_labels, labels_pred)*100:.1f}")
    print(f"Precision = {precision_score(test_labels, labels_pred, average='macro')*100:.1f}")
    print(f"Recall = {recall_score(test_labels, labels_pred, average='macro')*100:.1f}")
    print(f"F1 Score = {f1_score(test_labels, labels_pred, average='macro')*100:.1f}")
    print("\nDetailed classification report:")

    print(classification_report(test_labels, labels_pred, digits=3))

    cm = confusion_matrix(test_labels, labels_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()


###############################################################################################
# Helper Classes
###############################################################################################



###############################################################################################
# Main code
###############################################################################################

def main():
    train = pd.read_csv("wine_train.csv")
    test  = pd.read_csv("wine_test.csv")

    # Prints the first 5 rows of each dataset
    print("First 5 rows of wine_train:")
    print_df_info(train, num_of_rows=5, post_space=3)
    print(f"First 5 rows of wine_test:")
    print_df_info(test, num_of_rows=5, post_space=3)

    # Split each dataset to data and labels
    train_data, train_labels = split_data_label(train, 'target')
    test_data, test_labels = split_data_label(test, 'target')

    # Visuals
    if input("Would you like to see visual representations of the dataset for analysis? (y/n): ").strip().lower() == 'y':
        bar_chart(train_labels)
        pearson_heat_map(train_data)
        IQR_box_plot(train_data)
        PCA_scatter(train_data, train_labels)
        PCA_variance(train_data)
    else:
        print('skipping visualization')

    # Automated cross validation, using macro average f1 and 5 fold CV
    pipe_line = make_pipeline()
    param_grid = get_param_grid()
    grid = run_grid_search(pipe_line, param_grid,train_data, train_labels)
    summarize_grid_search(grid, 10)

    results_df = summarize_grid_search(grid)
    best_params = grid.best_params_
    chosen_num_of_componets = best_params['pca__n_components']
    chosen_c = best_params['clf__C']
    chosen_clf = best_params['clf']
    

    # reFitting the model with best params/classifier
    # Params   = {'clf': LogisticRegression(max_iter=2000, random_state=2025), 'clf__C': 0.01, 'pca__n_components': None}

    train_data_scaled = SCALER.fit_transform(train_data)
    
    final_PCA = PCA(n_components=chosen_num_of_componets, random_state=RANDOM)
    train_data_scaled_pca = final_PCA.fit_transform(train_data_scaled)

    clf_model = LogisticRegression(C = chosen_c, penalty= 'l2', solver = 'lbfgs', max_iter = 2000, random_state=RANDOM)
    clf_model.fit(train_data_scaled_pca, train_labels)

    data_test_scaled = SCALER.transform(test_data)
    data_test_pca = final_PCA.transform(data_test_scaled)
    labels_pred = clf_model.predict(data_test_pca)

    # Evaluation
    print_model_evaluation(test_labels, labels_pred)

if __name__ == "__main__":
    main()