# 🧤 Aero Glider (2022)

**Version 0.10** — Now with **word selection support!**

A hand-gesture based virtual keyboard and English word prediction system using Mediapipe + NLTK + OpenCV. Designed for gesture typing in mid-air without physical hardware.

![Main Interface](https://firebasestorage.googleapis.com/v0/b/kame-tech-lab.appspot.com/o/projects%2Faero-glider%2Fmain_image?alt=media&token=1a618011-a137-4686-a8b1-339459e7b1b7)

---

## 🎯 Overview

Aero Glider allows users to type English words using fingertip trajectories detected in real-time via webcam. It recognizes character zones mapped on a virtual keyboard and predicts possible words based on fuzzy string matching and density of key presses.

---

## ✨ Features

- 📷 **Hand Tracking with Mediapipe**
- 🎹 **Virtual keyboard overlay** using OpenCV
- 🧠 **Fuzzy word prediction** based on detected trajectory
- 📌 **2-letter substring filtering** for invalid patterns
- 🧮 **Outlier rejection** based on keyboard press densities
- 🔤 **English dictionary matching** via `nltk.words`
- ✅ **Word selection interface** (via 1/2/3 number keys)
- ⌨️ **Auto-sentence composition** on `Enter`

---

## 🧰 Tech Stack

- Python 3.x
- OpenCV (cv2)
- Mediapipe
- NLTK (words corpus)
- NumPy
- Difflib (SequenceMatcher)

---

## 🚀 How It Works

1. **Hand detection**: Tracks index fingertip (ID 8) when above knuckle (ID 5)
2. **Trajectory recording**: Accumulates positions & overlays path
3. **Key detection**: Maps coordinates to virtual keys based on pre-defined grid
4. **Key density counting**: High-count zones are interpreted as intended characters
5. **Filtering**: Removes outlier keypresses & invalid bi-grams
6. **Prediction**: Matches cleaned string to similar words in `nltk.words`
7. **Selection**: User chooses predicted word via keyboard (1/2/3)
8. **Output**: Appends selected word to composed sentence

---

## 📦 Keyboard Mapping

A QWERTY keyboard is mapped to a 10×3 grid. Example:

```
Q W E R T Y U I O P
A S D F G H J K L ;
Z X C V B N M , . /
```

---

## 🎮 Controls

- `1`, `2`, `3` → Select one of the 3 predicted words
- `Enter` → Confirm & append selected word to sentence
- `q` → Quit the application

---

## 📝 Notes

- Works best with good lighting & single hand input
- Virtual keyboard resizes to 800×600 window
- Real-time prediction results appear in bottom screen regions

---

## 📄 License

MIT License

---

Created by [@kamekingdom](https://github.com/kamekingdom)
