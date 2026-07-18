# 🍽️ NutriScan AI — Daily Calorie & Protein Tracker

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Model_Accuracy-78.62%25-green?style=flat-square)

An AI-powered food recognition and nutrition tracking web app built with **Deep Learning (MobileNetV2)** and **Transfer Learning**. Upload a food image to get instant calorie, protein, carb, and fat information, then track your daily nutrition goals.

---

## Project Status
- ✅ Phase 1 Completed
- ✅ Phase 2 Done

---

## 🎯 Features

- 📸 **Food Detection** — Detects 20 food categories from images using MobileNetV2
- 🔥 **Nutrition Tracking** — Calories, protein, carbs, and fat per meal
- 📊 **Daily Log** — Track all meals in one place
- 🎯 **Custom Goals** — Set personal calorie and protein targets
- 🏥 **Health Risk Analysis** — Get warnings based on daily intake
- 📈 **Progress Bars** — Visual progress toward daily goals

---

## 🧠 AI/ML Concepts Used

| Concept | Implementation |
| --- | --- |
| Transfer Learning | MobileNetV2 pretrained on ImageNet |
| CNN Architecture | Feature extraction from food images |
| Fine-tuning | Last 30 layers unfrozen for higher accuracy |
| Image Augmentation | Rotation, zoom, brightness, and flip |
| Softmax Classification | 20-class food prediction |
| Health Risk Prediction | Rule-based classifier on daily nutrition |

---

## 🍽️ Supported Food Categories

| | | | |
| --- | --- | --- | --- |
| 🍕 Pizza | 🍔 Hamburger | 🍣 Sushi | 🍜 Ramen |
| 🌮 Tacos | 🥞 Pancakes | 🧇 Waffles | 🍩 Donuts |
| 🍦 Ice Cream | 🍰 Cheesecake | 🍟 French Fries | 🌭 Hot Dog |
| 🥩 Steak | 🥗 Caesar Salad | 🍳 Omelette | 🍚 Fried Rice |
| 🍝 Spaghetti Bolognese | 🎂 Chocolate Cake | 🧀 Nachos | 🐟 Grilled Salmon |

---

## 📁 Project Structure

```text
calorie-tracker/
├── app/
│   └── app.py                  # Streamlit web app
├── notebooks/
│   ├── phase1_verify.ipynb     # Dataset verification
│   ├── phase2_model.ipynb      # MobileNetV2 training (101 classes)
│   ├── phase2_retrain.ipynb    # Retrained on 20 classes (78.62%)
│   └── phase3_nutrition.ipynb  # Nutrition pipeline
├── models/
│   ├── food_classifier_v2.h5   # Trained model
│   └── class_labels_v2.json    # Class label mapping
├── utils/
│   └── download_nutrition.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Prishu-Baranwal/calorie-tracker.git
cd calorie-tracker
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate    # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app/app.py
```

---

## 📊 Model Performance

| Metric | Value |
| --- | --- |
| Architecture | MobileNetV2 + Custom Head |
| Training Categories | 20 Food Classes |
| Dataset | Food-101 (subset) |
| Training Epochs | 20 + 10 (fine-tuning) |
| Final Validation Accuracy | **78.62%** |
| Base Model | ImageNet pretrained weights |

### Training Results

- Initial accuracy (101 classes): **52%**
- After retraining on 20 classes: **71%**
- After fine-tuning the last 30 layers: **78.62%** ✅

---


## 🔮 Future Improvements

- [ ] Add more food categories (50+)
- [ ] GPU training for higher accuracy
- [ ] Portion-size estimation using object detection
- [ ] Weekly nutrition reports
- [ ] Mobile app version

---

## 🛠️ Tech Stack

| Tool | Purpose |
| --- | --- |
| TensorFlow / Keras | Deep Learning model |
| MobileNetV2 | Transfer Learning backbone |
| OpenCV | Image preprocessing |
| Streamlit | Web app UI |
| Pandas / NumPy | Data processing |
| Scikit-learn | ML utilities |

---

## 👨‍💻 Author

**Prishu Baranwal**
B.Tech CSE | AI/ML Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-Prishu--Baranwal-black?style=flat-square&logo=github)](https://github.com/Prishu-Baranwal)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
