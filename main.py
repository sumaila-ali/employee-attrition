# Updated version preserving original formatting with 4 dynamically scaling charts, using CTk and restoring table-based metrics view

import customtkinter as ctk
import tkinter.filedialog as filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns


class ModelTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CustomTkinter Model Trainer")
        self.root.geometry("1400x1000")

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.df = None
        self.feature_names = []
        self.exclusion_list = []

        self.label_encoder = None
        self.X_test = None
        self.y_test = None
        self.model = None

        self.layout_ui()

    def layout_ui(self):
        self.load_button = ctk.CTkButton(self.root, text="Load CSV", command=self.load_csv)
        self.load_button.pack(pady=10)

        self.target_dropdown = ctk.CTkComboBox(self.root, values=[])
        self.target_dropdown.pack(pady=5)

        self.exclude_label = ctk.CTkLabel(self.root, text="Select columns to exclude from training")
        self.exclude_label.pack()

        self.exclude_listbox = ctk.CTkScrollableFrame(self.root, width=300, height=150)
        self.exclude_listbox.pack(pady=5)
        self.exclude_vars = {}

        self.model_var = ctk.StringVar(value="Logistic Regression")
        self.model_dropdown = ctk.CTkComboBox(self.root, values=["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"], variable=self.model_var)
        self.model_dropdown.pack(pady=5)

        self.scale_var = ctk.StringVar(value="None")
        self.scale_dropdown = ctk.CTkComboBox(self.root, values=["None", "StandardScaler", "MinMaxScaler"], variable=self.scale_var)
        self.scale_dropdown.pack(pady=5)

        self.smote_var = ctk.BooleanVar()
        self.smote_check = ctk.CTkCheckBox(self.root, text="Apply SMOTE", variable=self.smote_var)
        self.smote_check.pack(pady=5)

        self.train_button = ctk.CTkButton(self.root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=20)

        self.plot_frame = ctk.CTkFrame(self.root)
        self.plot_frame.pack(fill="both", expand=True)
        self.plot_frame.grid_rowconfigure((0, 1), weight=1)
        self.plot_frame.grid_columnconfigure((0, 1), weight=1)
        self.plot_grid = [[ctk.CTkFrame(self.plot_frame) for _ in range(2)] for _ in range(2)]

        for i in range(2):
            for j in range(2):
                self.plot_grid[i][j].grid(row=i, column=j, padx=5, pady=5, sticky="nsew")
                self.plot_grid[i][j].grid_rowconfigure(0, weight=1)
                self.plot_grid[i][j].grid_columnconfigure(0, weight=1)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            cols = self.df.columns.tolist()
            self.target_dropdown.configure(values=cols)
            self.target_dropdown.set(cols[0])
            self.exclude_vars = {}
            for widget in self.exclude_listbox.winfo_children():
                widget.destroy()
            for col in cols:
                var = ctk.BooleanVar()
                chk = ctk.CTkCheckBox(self.exclude_listbox, text=col, variable=var)
                chk.pack(anchor='w')
                self.exclude_vars[col] = var
            messagebox.showinfo("Info", "CSV loaded successfully.")

    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data first")
            return

        target = self.target_dropdown.get()
        self.exclusion_list = [col for col, var in self.exclude_vars.items() if var.get() and col != target]

        X = self.df.drop(columns=self.exclusion_list + [target])
        y = self.df[target]

        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        else:
            self.label_encoder = None

        X = pd.get_dummies(X, drop_first=True)
        self.feature_names = X.columns.tolist()

        scaler_choice = self.scale_var.get()
        if scaler_choice == "StandardScaler":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler_choice == "MinMaxScaler":
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if self.smote_var.get():
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        model_type = self.model_var.get()
        if model_type == "Logistic Regression":
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == "Decision Tree":
            self.model = DecisionTreeClassifier()
        elif model_type == "Random Forest":
            self.model = RandomForestClassifier()
        elif model_type == "XGBoost":
            self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(self.X_test)

        self.plot_metrics(self.y_test, y_pred, self.model, self.X_test, 0, 0)
        self.plot_cv_results(X_train, y_train, self.X_test, self.y_test, 0, 1)
        self.show_confusion_matrix(self.y_test, y_pred, 1, 0)
        self.show_feature_importance(1, 1)

    def plot_metrics(self, y_test, y_pred, model, X_test, row, col):
        # Evaluate metrics
        train_score = model.score(X_test, y_test)  # Using test set for simplicity
        test_score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Prepare metrics dictionary
        metrics = {
            "Model": model.__class__.__name__,
            "Training Score": train_score,
            "Test Score": test_score,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

        # Create a styled figure
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis('off')
        ax.set_facecolor('#ecf0f1')

        # Style settings
        title_font = {'fontsize': 20, 'fontweight': 'bold', 'color': '#2c3e50'}
        text_font = {'fontsize': 15, 'color': '#34495e'}

        # Title
        ax.text(0.5, 0.85, metrics["Model"], ha='center', **title_font, transform=ax.transAxes)

        # Metrics display
        spacing = 0.70
        for key in list(metrics.keys())[1:]:
            ax.text(0.5, spacing, f"{key}: {metrics[key]:.2f}", ha='center', **text_font, transform=ax.transAxes)
            spacing -= 0.10

        plt.tight_layout()

        # Show the figure directly in tkinter
        self.embed_plot(fig, row, col)

    def plot_cv_results(self, X_train, y_train, X_test, y_test, row, col):
        model_type = self.model_var.get()

        # Select model
        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_type == "Random Forest":
            model = RandomForestClassifier()
        elif model_type == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        else:
            return

        # 5-Fold Cross-Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        avg_cv = np.mean(cv_scores)
        std_cv = np.std(cv_scores)

        # Train and predict for classification report
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Get classification report
        target_names = (self.label_encoder.inverse_transform(np.unique(self.y_test)) 
                if self.label_encoder else list(map(str, np.unique(self.y_test))))
        report = classification_report(y_test, y_pred, target_names=target_names)
        report_lines = report.strip().split('\n')

        # Add 2-tab indent to the header line (assumes it's line 0)
        if len(report_lines) > 0:
            report_lines[0] = '        ' * 2 + report_lines[0]  # 16 spaces

        # Build plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.set_facecolor('#ecf0f1')

        # Fonts
        title_font = {'fontsize': 20, 'fontweight': 'bold', 'color': '#2c3e50'}
        summary_font = {'fontsize': 20, 'color': '#16a085'}
        text_font = {'fontsize': 20, 'color': '#34495e', 'family': 'monospace'}

        # Header
        ax.text(0.5, 1.02, f'{model_type} - CV and Classification Report', ha='center', **title_font, transform=ax.transAxes)
        ax.text(0.5, 0.97, f'5-Fold CV Avg Accuracy: {avg_cv:.4f} | Std Dev: {std_cv:.4f}', ha='center', **summary_font, transform=ax.transAxes)

        # Classification report lines
        y_pos = 0.88
        for line in report_lines:
            ax.text(0.01, y_pos, line, ha='left', **text_font, transform=ax.transAxes)
            y_pos -= 0.10

        plt.tight_layout()
        self.embed_plot(fig, row, col)

    def show_confusion_matrix(self, y_test, y_pred, row, col):
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        self.embed_plot(fig, row, col)

    def show_feature_importance(self, row, col):
        model_type = self.model_var.get()
        if model_type == "Logistic Regression" and hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        elif hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            importance = None

        if importance is not None:
            fi = pd.Series(importance, index=self.feature_names).sort_values(ascending=False).head(10)
            fig, ax = plt.subplots()
            fi.iloc[::-1].plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title("Top Feature Importances")
            self.embed_plot(fig, row, col)
        else:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Feature importance not available", ha='center')
            self.embed_plot(fig, row, col)

    def embed_plot(self, fig, row, col):
        for widget in self.plot_grid[row][col].winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_grid[row][col])
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == '__main__':
    root = ctk.CTk()
    app = ModelTrainerApp(root)
    root.mainloop()
