
# Hindi OCR - Handwritten Text to Digital 🖋️🇮🇳

> A Final Year Project developed under the guidance of the **National Informatics Centre (NIC)**.

This project demonstrates a deep learning-based Optical Character Recognition (OCR) system for handwritten **Devanagari words**. It aims to address the challenges of digitizing Indian handwritten documents by offering a web-based platform for real-time word recognition using a trained model.

---

## 🚀 Features

- ✅ Recognition of **handwritten Devanagari words** from images (JPG, PNG, JPEG).
- 🌐 Clean and responsive **web-based UI** built using HTML, CSS, and JavaScript.
- 🔍 Displays recognized text with **confidence scores**.
- 📂 Option to upload user images or test with **sample handwritten images**.
- 📱 Responsive layout with dark mode support.
- ⚠️ Error handling and loading indicators for better user experience.

---

## 🎯 Motivation & Objectives

India houses an extensive collection of handwritten records, literature, and government archives in **Devanagari and other Indic scripts**. Manual digitization is time-consuming and error-prone. Generic OCR models often fail to handle the unique complexities of Indian scripts such as **Shirorekha (headline), Matras (modifiers),** and **conjunct characters**.

### The key objectives of this project include:

- ✨ **Build a deep learning model** (CNN + BiLSTM + CTC) for Devanagari word recognition.
- 🌐 **Deploy a Flask-based web interface** for interactive predictions.
- 📊 **Evaluate model performance** using Character Error Rate (CER).
- 🏛️ Explore its **applicability in e-Governance** (Digital India, NIC initiatives, etc.).

---

## 🧠 Model Architecture

### CNN + BiLSTM + CTC Loss

1. Input grayscale image: **128x32**
2. 5-layer **Convolutional Neural Network (CNN)** for feature extraction.
3. Bidirectional **LSTM layers** for sequence learning.
4. **CTC Loss** for unsegmented word prediction.
5. Decoded output using greedy/best path CTC decoding.



**Training Dataset**:  
[IIT-H Devanagari Handwritten Word Dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data)

📈 **Final Character Error Rate (CER)**: `12.98%`  
🧪 Trained with additional custom samples to boost generalization.

---

## 🧪 Technology Stack

| Layer         | Technology                                      |
|---------------|-------------------------------------------------|
| Frontend      | HTML5, CSS3, Bootstrap 5, Vanilla JavaScript    |
| Backend       | Python 3.6, Flask                               |
| Deep Learning | TensorFlow 1.8.0, OpenCV, NumPy, Pillow, CTC    |
| Dataset       | IIIT-HW-Dev + custom handwritten word samples   |

---

## 📁 Project Structure

```
HindiOCR/
│
├── static/                # CSS, JS, Fonts
├── templates/             # Jinja2 HTML Templates (base.html, index.html, about.html)
├── model/                 # Trained model files, charList, snapshot
├── DataLoader.py          # Loads dataset and processes it for the model
├── Model.py               # CNN + LSTM + CTC architecture
├── SamplePreprocessor.py  # Image preprocessing and normalization
├── main.py                # CLI for training, validation, inference
├── upload.py              # Flask web server for frontend interaction
└── README.md              # This file
```

---

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/hindi-ocr-devnagari.git
cd hindi-ocr-devnagari
```

### 2. Install Dependencies

Make sure you have Python 3.6 and TensorFlow 1.8.0:

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python main.py --train
```

### 4. Validate the Model

```bash
python main.py --validate
```

### 5. Start the Web App

```bash
python upload.py
```

Visit `http://localhost:5000` in your browser.

---

## 🌐 Web Demo

> Upload a handwritten image in Devanagari script and view the prediction result instantly.

![Demo Video](https://drive.google.com/file/d/1mc3EnoepBATI9VV2maN8UsUSl2N9b3ay/view?usp=sharing)

---

## 🏛️ Relevance to NIC & Digital India

This project contributes to:

- 📂 **Digitization of handwritten records**
- ✍️ **Faster data entry from forms**
- 🔍 **Searchability of scanned documents**
- ♿ **Accessibility via screen readers and machine-readable output**

---

## 🔮 Future Work

- 📷 Improve preprocessing for noisy real-world images.
- 🔁 Apply data augmentation during training.
- 📚 Add better decoding via language models.
- 📜 Extend to **line-level recognition** and other Indic scripts (e.g., Nepali).

---

## 👩‍💻 Author

**Yashaswini Khansama**  
_Final Year MCA Student MATS University ,Raipur(C.G.)  
_Project guided by National Informatics Centre (NIC)
