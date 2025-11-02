# AI-Based Human Body Measurement API  
*(Tailoring & Fashion E-Commerce Ready)*

This project provides an **AI-powered body measurement API** built with **Flask**, **MediaPipe**, **OpenCV**, and **PyTorch**.  
By analyzing **front and side images**, it estimates key human body measurements for **tailoring, virtual fitting, and fashion e-commerce** — without requiring manual measuring tools or external APIs.

---

## ✨ Key Features

- **Full-body image measurement** via front + optional side pose.
- **AI depth estimation** using a **local MiDaS model** (no external downloads).
- **Automatic scale calibration** via user height or A4 reference object.
- **JSON response with measurements in cm and inches**.
- **Quality feedback** — API returns human-readable notes if measurements seem off (e.g. hips too wide, waist too small).
- **Lightweight Docker + Render deployment**.
- **Runs fully offline** (no external API dependencies).

---

## 🧠 Core Libraries

| Library | Purpose |
|----------|----------|
| `Flask` | REST API server |
| `MediaPipe` | Pose landmark detection (shoulders, hips, knees, ankles) |
| `OpenCV` | Image preprocessing and contour analysis |
| `PyTorch` | Local **MiDaS** model for depth inference |
| `Gunicorn` | Production WSGI server (used on Render) |

---

## ⚙️ How It Works

1. **Pose Detection**  
   MediaPipe finds 3D body landmarks (shoulders, hips, ankles, etc.).

2. **Scale Calibration**  
   The system computes pixel-to-cm ratio using either:
   - User-provided `height_cm`, or  
   - A detected A4 reference sheet.

3. **Depth Estimation**  
   Local MiDaS model produces a depth map to improve circumference accuracy.

4. **Measurement Extraction**  
   Uses geometric approximations (elliptical model) to calculate chest, waist, hip, neck, thigh, and limb dimensions.

5. **Validation & Feedback**  
   If proportions are unrealistic, the system adds `"notes"` to the JSON output suggesting how to retake the image.

---

## 🚀 API Overview

### **POST** `/upload_images`

**Form fields:**

| Field | Type | Required | Description |
|--------|------|-----------|-------------|
| `front` | file | ✅ | Full-body front image |
| `left_side` | file | Optional | Side image for better depth estimation |
| `height_cm` | float | Optional | User height (improves scaling) |

---

### ✅ Example Request

```bash
curl -X POST https://live-measurements-api.onrender.com/upload_images \
  -F "front=@front.jpeg" \
  -F "left_side=@left_side.jpeg" \
  -F "height_cm=170"
### example response
{
  "debug_info": {
    "focal_length": 3514.28,
    "scale_factor": 0.1707,
    "user_height_cm": 170.0
  },
  "measurements": {
    "shoulder_width_cm": 42.68,
    "chest_circumference_cm": 111.54,
    "chest_circumference_in": 43.92,
    "waist_circumference_cm": 47.19,
    "waist_circumference_in": 18.58,
    "hip_circumference_cm": 202.32,
    "hip_circumference_in": 79.66,
    "notes": [
      "Hip reading may be distorted by clothing, stance, or camera angle. Stand straight, keep feet under hips, arms slightly out, and avoid jackets or loose fabric.",
      "Waist reading may be too small. Make sure your midsection is fully visible and not heavily shadowed or blocked by arms."
    ]
  }
}


> 📌 **Note:**  
> The system uses **AI depth maps** and **contour-based width detection**.  
> Final measurements may have a **±2–3 cm variance** depending on image quality and user alignment.


# Integration in Fashion E-Commerce

This solution is plug-and-play for:

- **E-commerce brands** offering size suggestions or virtual try-ons.
- **Tailoring platforms** wanting remote client measurements.
- **Clothing manufacturers** personalizing size charts for customers.
- **Fashion mobile apps** for custom-fitted clothing suggestions.

Simply integrate this API into your frontend — mobile or web — to collect two photos and retrieve exact measurements.


## 🤝 Contributions

PRs and suggestions are welcome! Fork this repo, raise an issue, or open a pull request.

## 📜 License

MIT License. Feel free to use this for personal or commercial projects — just give credit.



